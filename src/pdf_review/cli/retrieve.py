"""Build retrieval evidence (text + visual page scoring) for a local PDF.

CLI wrapper. Produces evidence.json for the 'generate' CLI's strict
extraction, evidence_ref-based reviewer, and audit. Logic lives in the core
pdf_review package.

Note: this script previously also had a --retrieval-only flag and a legacy
extraction/reviewer/audit path used when it was omitted. That path predated
the strict validators and was never used in practice (the 'generate' CLI was
always run afterwards instead); it was removed once a structural diff
against real production output (document_stats.json, extracted record
counts, audit.json citation_checks) confirmed this script's retrieval output
was unaffected. This script now only ever produces evidence.json.
"""
import argparse
import gc
import hashlib
from pathlib import Path

from ..constants import TOPICS
from ..io import atomic_json, save
from ..pdf import extract_pages, render_page_images
from ..retrieval import retrieve_and_rerank, select_diverse_chunks
from ..segmentation import build_chunks
from ..visual_retrieval import score_pages_visually
from ..vlm import VLMRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--vlm-model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--chunk-tokens", type=int, default=240)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--evidence-chunks", type=int, default=6)
    parser.add_argument(
        "--max-chunks-per-page",
        type=int,
        default=2,
        help=(
            "Cap how many of the top evidence_chunks may come from the same page, so "
            "overlapping windows of one section (e.g. the introduction) don't crowd out "
            "a lower-ranked but distinct-page chunk. Set >= evidence-chunks to disable."
        ),
    )
    parser.add_argument("--image-pages", type=int, default=3)
    parser.add_argument("--visual-batch-pages", type=int, default=2)
    parser.add_argument("--visual-max-new-tokens", type=int, default=256)
    parser.add_argument("--json-retries", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument(
        "--segmentation",
        choices=("page", "semantic"),
        default="semantic",
        help="Use physical pages or detected document headings as section boundaries",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=tuple(TOPICS),
        default=list(TOPICS),
        help="Review only the selected topics",
    )
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not 0 <= args.overlap_tokens < args.chunk_tokens:
        parser.error("Require 0 <= overlap-tokens < chunk-tokens")
    if args.evidence_chunks < 1:
        parser.error("--evidence-chunks must be at least 1")
    if args.max_chunks_per_page < 1:
        parser.error("--max-chunks-per-page must be at least 1")
    if args.retrieve_k < args.evidence_chunks:
        parser.error("--retrieve-k must be at least --evidence-chunks")
    if args.image_pages < 1 or args.visual_batch_pages < 1:
        parser.error("Image page counts must be at least 1")
    if args.json_retries < 0:
        parser.error("--json-retries cannot be negative")
    return args


# Retry hint for visual page scoring's generate_json calls, reproducing the
# original run_general_pdf_review.py's single shared generate_json() wording.
_VISUAL_SCORING_RETRY_HINT = (
    "Keep all text fields concise and complete the JSON within the output limit."
)


def _config_generation_conditions():
    return {
        "text_retrieval_text_only": "Text-retrieved chunks and their extracted page text; no visual retrieval pages or pixels",
        "hybrid_retrieval_text_only": "Text evidence plus extracted text from visually retrieved pages; no pixels",
        "hybrid_retrieval_with_images": "The same hybrid evidence plus pixels from visually retrieved pages",
    }


def main():
    args = parse_args()
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this script in a GPU Slurm job.")
    args.output_dir.mkdir(parents=True)
    selected_topics = {name: TOPICS[name] for name in args.topics}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["pdf_sha256"] = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    config["topic_definitions"] = selected_topics
    config["text_selection"] = "Top chunks per subquery after dense retrieval and cross-encoder reranking"
    config["visual_selection"] = (
        "Whole-document page-image relevance scoring with Qwen2.5-VL; "
        "text reranker page rank breaks equal visual scores"
    )
    config["generation_conditions"] = _config_generation_conditions()
    save(args.output_dir / "config.json", config)

    embedder = SentenceTransformer(args.embedding_model, device="cuda", local_files_only=True)
    tokenizer = embedder.tokenizer
    if args.chunk_tokens + tokenizer.num_special_tokens_to_add(pair=False) > embedder.max_seq_length:
        raise ValueError("Chunk size exceeds embedding model input limit")

    pages, layout_page_texts = extract_pages(args.pdf)
    save(args.output_dir / "pages.json", pages)
    layout_text_path = args.output_dir / "layout_text.txt"
    layout_text_path.write_text("\n\f\n".join(layout_page_texts), encoding="utf-8")
    nonempty_pages = sum(bool(page["text"].strip()) for page in pages)
    if not pages:
        raise ValueError("The PDF contains no pages")
    if nonempty_pages == 0:
        raise ValueError(
            "The PDF has no extractable text layer. Run OCR before using this reviewer."
        )

    chunks, sections = build_chunks(pages, tokenizer, args.segmentation, args.chunk_tokens, args.overlap_tokens)
    if args.segmentation == "semantic":
        save(args.output_dir / "sections.json", sections)
    save(args.output_dir / "chunks.json", chunks)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF")
    atomic_json(args.output_dir / "document_stats.json", {
        "page_count": len(pages),
        "nonempty_text_page_count": nonempty_pages,
        "empty_text_page_count": len(pages) - nonempty_pages,
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "segmentation": args.segmentation,
    })

    chunk_vectors = embedder.encode([item["text"] for item in chunks], batch_size=args.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True)
    all_queries = [query for topic in selected_topics.values() for query in topic["subqueries"]]
    query_vectors = embedder.encode(all_queries, batch_size=args.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True)
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    reranker = CrossEncoder(args.reranker_model, device="cuda", local_files_only=True)
    evidence = {}
    offset = 0
    for topic_name, topic in selected_topics.items():
        traces = []
        for query in topic["subqueries"]:
            ranked = retrieve_and_rerank(
                query, query_vectors[offset], chunk_vectors, chunks, reranker,
                args.retrieve_k, args.batch_size,
            )
            offset += 1
            query_selected = select_diverse_chunks(ranked, args.evidence_chunks, args.max_chunks_per_page)
            trace = [
                {
                    "chunk_id": item["chunk_id"],
                    "section_id": item.get("section_id"),
                    "section_title": item.get("section_title"),
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "reranker_rank": item["reranker_rank"],
                    "reranker_score": item["reranker_score"],
                }
                for item in query_selected
            ]
            query_pages = []
            for item in query_selected:
                for page_number in range(item["page_start"], item["page_end"] + 1):
                    if page_number not in query_pages:
                        query_pages.append(page_number)
            traces.append({
                "subquery": query,
                "top_candidates": trace,
                "selected_text": query_selected,
                "selected_text_pages": query_pages,
                "selected_image_pages": [],
                "visual_scores": [],
            })
        evidence[topic_name] = {"subquery_traces": traces}
    del reranker, chunk_vectors, query_vectors
    gc.collect()
    torch.cuda.empty_cache()

    image_paths = render_page_images(args.pdf, args.output_dir, len(pages))

    runner = VLMRunner(args.vlm_model, args.max_pixels)

    def generate_json(messages, max_new_tokens, validator):
        return runner.generate_json(
            messages, max_new_tokens, validator,
            retry_hint=_VISUAL_SCORING_RETRY_HINT,
            max_retries=args.json_retries,
            output_dir=args.output_dir,
            keep_parsed_if_validator_returns_none=True,
        )

    score_pages_visually(
        evidence, pages, image_paths, generate_json,
        args.visual_batch_pages, args.visual_max_new_tokens, args.image_pages, args.output_dir,
    )
    atomic_json(args.output_dir / "evidence.json", evidence)
    print(f"Saved retrieval evidence: {args.output_dir / 'evidence.json'}", flush=True)
    print(
        "Run the 'generate' CLI with --evidence pointing at this file for strict "
        "extraction, reviewer, and audit.",
        flush=True,
    )


if __name__ == "__main__":
    main()
