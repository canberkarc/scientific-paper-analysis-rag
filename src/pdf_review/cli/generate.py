"""Rerun strict extraction and review generation from an existing evidence.json.

CLI wrapper. Ported from rerun_general_pdf_generation.py with identical
behavior; logic lives in the core pdf_review package.
"""
import argparse
import json
from pathlib import Path

from ..audit import (
    build_audit,
    build_citation_checks,
    build_condition_comparison,
    build_normalized_page_text,
    build_page_concentration_diagnostics,
)
from ..constants import GENERATION_MODES, TOPICS
from ..extraction import (
    EXTRACTION_RETRY_HINT,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_schema,
    render_records,
    validate_extraction,
)
from ..io import atomic_json
from ..pdf import extract_pages, render_page_images, select_layout_excerpt
from ..reviewer import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_TASK_PROMPT_PREFIX,
    build_reviewer_input,
    build_reviewer_schema,
    render_reviewer,
    validate_reviewer,
)
from ..validation import validate_reused_evidence
from ..vlm import VLMRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vlm-model",
        default="Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--reviewer-max-new-tokens", type=int, default=1024)
    parser.add_argument("--json-retries", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=tuple(TOPICS),
        default=list(TOPICS),
    )
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if not args.evidence.is_file():
        parser.error(f"Evidence file not found: {args.evidence}")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.json_retries < 0:
        parser.error("--json-retries cannot be negative")
    return args


def main():
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run this script in a GPU Slurm job.")

    selected_topics = {name: TOPICS[name] for name in args.topics}
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    pages, layout_page_texts = extract_pages(args.pdf)
    if not pages:
        raise ValueError("The PDF contains no pages")
    if not any(page["text"].strip() for page in pages):
        raise ValueError("The PDF has no extractable text layer")
    validate_reused_evidence(evidence, selected_topics, len(pages))

    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "config.json", {
        "pdf": str(args.pdf),
        "evidence": str(args.evidence),
        "vlm_model": args.vlm_model,
        "max_new_tokens": args.max_new_tokens,
        "reviewer_max_new_tokens": args.reviewer_max_new_tokens,
        "json_retries": args.json_retries,
        "max_pixels": args.max_pixels,
        "topics": list(selected_topics),
        "pipeline": "strict_generation_from_reused_evidence",
    })
    atomic_json(args.output_dir / "pages.json", pages)
    atomic_json(args.output_dir / "evidence.json", evidence)
    (args.output_dir / "layout_text.txt").write_text(
        "\n\f\n".join(layout_page_texts), encoding="utf-8"
    )
    image_paths = render_page_images(args.pdf, args.output_dir, len(pages))

    runner = VLMRunner(args.vlm_model, args.max_pixels)

    def generate_json(messages, max_new_tokens, validator):
        # Reviewer generation reuses this same retry hint, exactly as the
        # original script's one shared generate_json() closure did.
        return runner.generate_json(
            messages, max_new_tokens, validator,
            retry_hint=EXTRACTION_RETRY_HINT,
            max_retries=args.json_retries,
            output_dir=args.output_dir,
        )

    results = []
    intermediate_extractions = {}
    for topic_name, topic in selected_topics.items():
        answers = {}
        for mode in GENERATION_MODES:
            structured_records = []
            for trace in evidence[topic_name]["subquery_traces"]:
                subquery = trace["subquery"]
                visual_pages = (
                    trace["selected_image_pages"]
                    if mode != "text_retrieval_text_only"
                    else []
                )
                allowed_pages = list(dict.fromkeys(
                    trace["selected_text_pages"] + visual_pages
                ))
                context_parts = []
                for item in trace["selected_text"]:
                    page_label = (
                        str(item["page_start"])
                        if item["page_start"] == item["page_end"]
                        else f"{item['page_start']}-{item['page_end']}"
                    )
                    context_parts.append(
                        f"[PDF page {page_label}; chunk {item['chunk_id']}]\n{item['text']}"
                    )
                text_pages = set(trace["selected_text_pages"])
                for page_number in visual_pages:
                    if page_number not in text_pages and pages[page_number - 1]["text"]:
                        context_parts.append(
                            f"[PDF page {page_number}; extracted page text]\n"
                            f"{pages[page_number - 1]['text']}"
                        )
                for page_number in allowed_pages:
                    excerpt = select_layout_excerpt(
                        layout_page_texts[page_number - 1]
                    )
                    if excerpt:
                        context_parts.append(
                            f"[PDF page {page_number}; layout-preserved excerpt]\n{excerpt}"
                        )

                content = []
                if mode == "hybrid_retrieval_with_images":
                    for page_number in visual_pages:
                        content.extend([
                            {"type": "text", "text": f"PDF page {page_number}:"},
                            {"type": "image", "image": image_paths[page_number]},
                        ])
                schema = build_extraction_schema(allowed_pages)
                allowed_page_strings = {str(page) for page in allowed_pages}
                task = (
                    f"Question: {subquery}\nReturn JSON only with this schema: "
                    f"{json.dumps(schema)}. The only allowed pdf_page values are: "
                    f"{', '.join(map(str, allowed_pages))}. Use actual JSON null for "
                    "inapplicable fields. The value field must contain only a numerical value "
                    "without its unit. If the evidence does not support an answer, return an "
                    "empty records array and explain briefly in unsupported."
                )
                content.append({
                    "type": "text",
                    "text": (
                        f"Evidence:\n\n{'\n\n'.join(context_parts)}\n\n"
                        f"Extraction task:\n{task}"
                    ),
                })

                parsed, raw, counts, attempts = generate_json(
                    [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    args.max_new_tokens,
                    lambda value: validate_extraction(value, allowed_page_strings),
                )
                structured_records.append({
                    "subquery": subquery,
                    "allowed_pages": allowed_pages,
                    "record": parsed,
                    "raw_record": raw,
                    "failed_attempts": attempts,
                    **counts,
                })
                intermediate_extractions.setdefault(topic_name, {})[mode] = (
                    structured_records
                )
                atomic_json(
                    args.output_dir / "intermediate_extractions.json",
                    intermediate_extractions,
                )
            answers[mode] = {
                "answer": render_records(topic_name, structured_records),
                "input_tokens": sum(item["input_tokens"] for item in structured_records),
                "generated_tokens": sum(
                    item["generated_tokens"] for item in structured_records
                ),
                "assembly": "deterministic_from_strict_validated_extractions",
            }
        results.append({
            "topic": topic_name,
            "subqueries": topic["subqueries"],
            "retrieval": evidence[topic_name]["subquery_traces"],
            "answers": answers,
        })
        atomic_json(args.output_dir / "review_results.json", {
            "completed_topics": len(results),
            "expected_topics": len(selected_topics),
            "requested_topics": list(selected_topics),
            "results": results,
        })
        print(f"Completed topic: {topic_name}", flush=True)

    reviewer_results = {}
    for mode in GENERATION_MODES:
        reviewer_input, evidence_lookup = build_reviewer_input(
            intermediate_extractions, selected_topics, mode
        )
        allowed_refs = set(evidence_lookup)
        # Deterministic choice (sorted, not next(iter(set))): the schema example
        # embedded in the prompt must not vary across process runs — an earlier
        # bug picked an arbitrary element via Python's per-process string hash
        # randomization, which changed the prompt (and hence reviewer output)
        # between otherwise-identical runs. Fixed and verified byte-identical
        # across independent runs.
        example_ref = sorted(allowed_refs)[0] if allowed_refs else None
        schema = build_reviewer_schema(example_ref)
        prompt = (
            f"{REVIEWER_TASK_PROMPT_PREFIX}{json.dumps(schema)}\nStructured evidence:\n"
            f"{json.dumps(reviewer_input, ensure_ascii=False)}"
        )

        parsed, raw, counts, attempts = generate_json(
            [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            args.reviewer_max_new_tokens,
            lambda value: validate_reviewer(value, allowed_refs, evidence_lookup),
        )
        reviewer_results[mode] = {
            "structured": parsed,
            "review": render_reviewer(parsed),
            "raw_response": raw,
            "failed_attempts": attempts,
            **counts,
        }
        atomic_json(
            args.output_dir / "reviewer_assessment.json", reviewer_results
        )

    normalized_page_text = build_normalized_page_text(pages, layout_page_texts)
    citation_checks = build_citation_checks(
        selected_topics, GENERATION_MODES, intermediate_extractions, normalized_page_text
    )
    audit = build_audit(citation_checks)
    atomic_json(args.output_dir / "audit.json", audit)

    page_diagnostics = build_page_concentration_diagnostics(
        selected_topics, GENERATION_MODES, intermediate_extractions
    )
    atomic_json(
        args.output_dir / "condition_comparison.json",
        build_condition_comparison(
            selected_topics, GENERATION_MODES, intermediate_extractions,
            reviewer_results, page_diagnostics,
        ),
    )
    output = {
        "completed_topics": len(results),
        "expected_topics": len(selected_topics),
        "requested_topics": list(selected_topics),
        "results": results,
        "reviewer_assessment": reviewer_results,
        "audit": audit,
    }
    atomic_json(args.output_dir / "review_results.json", output)
    print(f"Saved: {args.output_dir / 'review_results.json'}", flush=True)


if __name__ == "__main__":
    main()
