"""Regression tests for retrieval.py's page-diversity selection.

Real failure this guards against: 5 of 6 top-reranked chunks all came from
the same overlapping-window page (introduction/related-work), crowding out
the page that actually contained the paper's proposed method.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.retrieval import select_diverse_chunks


def _chunk(page):
    return {"page_start": page, "page_end": page, "chunk_id": f"p{page}"}


def test_page_cap_prevents_one_page_from_dominating():
    # 5 overlapping windows of page 2, 1 of page 3, 1 of page 5 (the one that matters)
    ranked = [
        (_chunk(2), 0.9), (_chunk(2), 0.85), (_chunk(2), 0.8),
        (_chunk(2), 0.75), (_chunk(2), 0.7), (_chunk(3), 0.65),
        (_chunk(5), 0.1),
    ]
    selected = select_diverse_chunks(ranked, keep=4, max_chunks_per_page=2)
    pages = [item["page_start"] for item in selected]
    assert pages.count(2) <= 2
    assert 5 in pages  # the low-ranked-but-distinct page must get a slot


def test_reranker_rank_and_score_attached_in_order():
    ranked = [(_chunk(1), 0.9), (_chunk(2), 0.5)]
    selected = select_diverse_chunks(ranked, keep=2, max_chunks_per_page=2)
    assert selected[0]["reranker_rank"] == 1
    assert selected[0]["reranker_score"] == 0.9
    assert selected[1]["reranker_rank"] == 2


def test_keep_limit_respected():
    ranked = [(_chunk(page), 1.0 - page * 0.1) for page in range(1, 10)]
    selected = select_diverse_chunks(ranked, keep=3, max_chunks_per_page=1)
    assert len(selected) == 3
