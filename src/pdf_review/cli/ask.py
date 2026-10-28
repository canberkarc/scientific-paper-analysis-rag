"""Answer one ad-hoc question about a PDF, citing supporting pages.

CLI wrapper. Ported from ask_pdf_question.py with identical behavior. Reuses
a prior 'retrieve --retrieval-only' run's cached chunks.json/pages.json/
page_*.jpg, so it only needs to embed and rerank for the new question, then
ask the VLM for a cited answer. This is separate from the fixed 5-topic/
10-subquery review pipeline: it answers whatever the user actually asks,
using the same strict JSON schema, page-diversity selection, and every
normalization fix built for the main pipeline.

Usage:
    python -m pdf_review.cli.ask \
        --evidence-dir general_pdf_review_regression_v9_evidence \
        --question "What forgetting factor value is used for the Kalman filter?" \
        --output-dir general_pdf_review_question_1

Add --with-images to also give the VLM the page images for the retrieved
pages (slower, sometimes more accurate for content only visible in figures
or tables).
"""
import argparse
import gc
import json
from pathlib import Path

from ..io import atomic_json
from ..question import ANSWER_RETRY_HINT, build_answer_system_prompt, render_answer, validate_answer
from ..retrieval import retrieve_and_rerank, select_diverse_chunks
from ..vlm import VLMRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir", type=Path, required=True,
        help="Output directory of a prior 'retrieve --retrieval-only' run "
             "(must contain chunks.json and pages.json).",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--vlm-model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--answer-chunks", type=int, default=8)
    parser.add_argument("--max-chunks-per-page", type=int, default=2)
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--image-pages", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--json-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not (args.evidence_dir / "chunks.json").is_file():
        parser.error(
            f"{args.evidence_dir}/chunks.json not found; run "
            "the 'retrieve' CLI with --retrieval-only on this PDF first."
        )
    if not (args.evidence_dir / "pages.json").is_file():
        parser.error(f"{args.evidence_dir}/pages.json not found")
    if not args.question.strip():
        parser.error("--question cannot be empty")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.answer_chunks < 1:
        parser.error("--answer-chunks must be at least 1")
    if args.retrieve_k < args.answer_chunks:
        parser.error("--retrieve-k must be at least --answer-chunks")
    if args.max_chunks_per_page < 1:
        parser.error("--max-chunks-per-page must be at least 1")
    if args.image_pages < 1:
        parser.error("--image-pages must be at least 1")
    if args.json_retries < 0:
        parser.error("--json-retries cannot be negative")
    return args


def main():
    args = parse_args()
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this script in a GPU Slurm job.")

    chunks = json.loads((args.evidence_dir / "chunks.json").read_text(encoding="utf-8"))
    pages = json.loads((args.evidence_dir / "pages.json").read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(f"{args.evidence_dir}/chunks.json is empty")

    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "config.json", {
        "evidence_dir": str(args.evidence_dir),
        "question": args.question,
        "vlm_model": args.vlm_model,
        "answer_chunks": args.answer_chunks,
        "max_chunks_per_page": args.max_chunks_per_page,
        "with_images": args.with_images,
        "json_retries": args.json_retries,
        "pipeline": "ad_hoc_question",
    })

    embedder = SentenceTransformer(args.embedding_model, device="cuda", local_files_only=True)
    chunk_vectors = embedder.encode(
        [chunk["text"] for chunk in chunks], batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    query_vector = embedder.encode(
        [args.question], normalize_embeddings=True, convert_to_numpy=True,
    )[0]
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    reranker = CrossEncoder(args.reranker_model, device="cuda", local_files_only=True)
    ranked = retrieve_and_rerank(
        args.question, query_vector, chunk_vectors, chunks, reranker,
        args.retrieve_k, args.batch_size,
    )
    del reranker
    gc.collect()
    torch.cuda.empty_cache()

    selected = select_diverse_chunks(ranked, args.answer_chunks, args.max_chunks_per_page)
    if not selected:
        raise ValueError("No chunks were selected; check --answer-chunks and --retrieve-k")

    selected_pages = []
    for item in selected:
        for page_number in range(item["page_start"], item["page_end"] + 1):
            if page_number not in selected_pages:
                selected_pages.append(page_number)
    allowed_page_strings = {str(page_number) for page_number in selected_pages}

    atomic_json(args.output_dir / "retrieval_trace.json", {
        "question": args.question,
        "selected_chunks": selected,
        "selected_pages": selected_pages,
    })

    evidence_text = "\n\n".join(
        f"[PDF page {item['page_start']}] {item['text']}" for item in selected
    )
    system = build_answer_system_prompt(selected_pages)
    content = [{
        "type": "text",
        "text": f"Question: {args.question}\n\nStructured evidence:\n{evidence_text}",
    }]
    if args.with_images:
        added = 0
        for page_number in selected_pages:
            if added >= args.image_pages:
                break
            image_path = args.evidence_dir / f"page_{page_number:04d}.jpg"
            if image_path.is_file():
                content.insert(0, {"type": "image", "image": f"file://{image_path.resolve()}"})
                added += 1

    runner = VLMRunner(args.vlm_model, args.max_pixels)
    parsed, raw, counts, attempts = runner.generate_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        args.max_new_tokens,
        lambda value: validate_answer(value, allowed_page_strings, selected_pages),
        retry_hint=ANSWER_RETRY_HINT,
        max_retries=args.json_retries,
        output_dir=args.output_dir,
    )

    answer_text = render_answer(parsed, len(selected), selected_pages)

    atomic_json(args.output_dir / "answer.json", {
        "question": args.question,
        "answer": parsed,
        "answer_text": answer_text,
        "selected_pages": selected_pages,
        "raw_response": raw,
        "failed_attempts": attempts,
        **counts,
    })
    print(f"Saved: {args.output_dir / 'answer.json'}", flush=True)
    print("\n" + answer_text, flush=True)


if __name__ == "__main__":
    main()
