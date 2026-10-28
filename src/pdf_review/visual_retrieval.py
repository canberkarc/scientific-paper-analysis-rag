"""Whole-page visual relevance scoring for each retrieved subquery.

Ported unchanged from run_general_pdf_review.py's main(). generate_json is
injected as a callable (bound to a loaded VLMRunner in the CLI) instead of
being a closure over module-level model/processor/args state.
"""
import json

from .io import atomic_json
from .normalization import normalize_pdf_page, normalize_relevance


def score_pages_visually(
    evidence, pages, image_paths, generate_json,
    visual_batch_pages, visual_max_new_tokens, image_pages, output_dir,
):
    """Mutate `evidence` in place: add visual_scores/selected_image_pages/
    visual_selection_source to every subquery trace. Checkpoints to
    output_dir/evidence_checkpoint.json exactly as often as the original.
    """
    for topic in evidence.values():
        for trace in topic["subquery_traces"]:
            scores = []
            page_numbers = list(range(1, len(pages) + 1))
            for start in range(0, len(page_numbers), visual_batch_pages):
                batch_pages = page_numbers[start:start + visual_batch_pages]
                content = []
                for image_index, page_number in enumerate(batch_pages, 1):
                    content.extend([
                        {
                            "type": "text",
                            "text": f"Candidate image {image_index} is PDF page {page_number}:",
                        },
                        {"type": "image", "image": image_paths[page_number]},
                    ])
                expected_pages = {str(page_number) for page_number in batch_pages}
                visual_schema = {
                    "scores": [
                        {
                            "pdf_page": page_number,
                            "relevance": 0,
                            "reason": "brief visual reason",
                        }
                        for page_number in batch_pages
                    ]
                }
                content.append({
                    "type": "text",
                    "text": (
                        f"Question: {trace['subquery']}\n"
                        "Score the visual usefulness of every candidate page for answering the question. "
                        "Judge figures, tables, diagrams, equations, and visibly structured evidence. "
                        f"Return exactly one score for each PDF page {batch_pages}. "
                        f"Use these exact pdf_page values and this JSON structure: "
                        f"{json.dumps(visual_schema)}. Keep each reason to at most 10 words. "
                        "Relevance must be an integer: "
                        "3 directly useful, 2 strongly relevant, 1 weakly relevant, 0 irrelevant."
                    ),
                })

                def validate_visual(value):
                    if isinstance(value, list):
                        value = {"scores": value}
                    elif (
                        isinstance(value, dict)
                        and "scores" not in value
                        and "pdf_page" in value
                        and "relevance" in value
                    ):
                        value = {"scores": [value]}
                    if not isinstance(value, dict) or not isinstance(value.get("scores"), list):
                        raise ValueError("Visual scores must be a JSON object containing a scores list")
                    returned = set()
                    for row in value["scores"]:
                        page = normalize_pdf_page(row.get("pdf_page", ""), expected_pages)
                        relevance = normalize_relevance(row.get("relevance"))
                        if page not in expected_pages:
                            raise ValueError(
                                f"Unexpected visual score page: {page}. "
                                f"Allowed pages are {sorted(expected_pages, key=int)}"
                            )
                        if not isinstance(relevance, int) or relevance not in (0, 1, 2, 3):
                            raise ValueError(f"Invalid visual relevance for page {page}: {relevance}")
                        row["pdf_page"] = page
                        row["relevance"] = relevance
                        returned.add(page)
                    if returned != expected_pages or len(value["scores"]) != len(batch_pages):
                        raise ValueError(
                            "Exactly one visual relevance score is required for each candidate "
                            f"PDF page {sorted(expected_pages, key=int)}"
                        )
                    return value

                parsed, raw, counts, attempts = generate_json(
                    [{"role": "system", "content": (
                        "You rank scientific PDF page images for visual evidence retrieval. "
                        "Use the visible page only and return the requested JSON."
                    )}, {"role": "user", "content": content}],
                    visual_max_new_tokens,
                    validate_visual,
                )
                for row in parsed["scores"]:
                    scores.append({
                        "pdf_page": int(row["pdf_page"]),
                        "relevance": int(row["relevance"]),
                        "reason": str(row.get("reason", "")).strip(),
                        "raw_response": raw,
                        "input_tokens": counts["input_tokens"],
                        "generated_tokens": counts["generated_tokens"],
                        "failed_attempts": attempts,
                    })
                trace["visual_scores"] = scores
                atomic_json(output_dir / "evidence_checkpoint.json", evidence)
            text_page_rank = {
                page_number: rank
                for rank, item in enumerate(trace["selected_text"], 1)
                for page_number in range(item["page_start"], item["page_end"] + 1)
                if page_number not in {
                    page
                    for earlier in trace["selected_text"][:rank - 1]
                    for page in range(earlier["page_start"], earlier["page_end"] + 1)
                }
            }
            ranked_visual = sorted(scores, key=lambda row: (
                -row["relevance"],
                text_page_rank.get(row["pdf_page"], len(pages) + 1),
                row["pdf_page"],
            ))
            selected_visual = [
                row["pdf_page"] for row in ranked_visual
                if row["relevance"] > 0
            ][:image_pages]
            if not selected_visual:
                selected_visual = trace["selected_text_pages"][:image_pages]
                trace["visual_selection_source"] = "text_page_fallback_no_positive_visual_score"
            else:
                trace["visual_selection_source"] = "positive_visual_score"
            trace["visual_scores"] = scores
            trace["selected_image_pages"] = selected_visual
            atomic_json(output_dir / "evidence_checkpoint.json", evidence)
