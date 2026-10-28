"""Dense retrieval, reranking, and page-diverse chunk selection.

retrieve_and_rerank unifies the identical top-k-then-rerank mechanics that
were duplicated between run_general_pdf_review.py's per-subquery retrieval
loop and ask_pdf_question.py's single-question retrieval.

select_diverse_chunks is ask_pdf_question.py's page-diversity-cap function,
unchanged; the same logic was previously inlined (not shared) in
run_general_pdf_review.py's retrieval loop. Callers that also need the
lightweight "trace" summary (run_general_pdf_review.py's evidence.json
top_candidates) build it from the returned items' reranker_rank/score,
which reproduces the original inline trace exactly (see cli/retrieve.py).
"""
import numpy as np


def retrieve_and_rerank(query_text, query_vector, chunk_vectors, chunks, reranker, retrieve_k, batch_size):
    """Top-k dense retrieval by cosine score, then cross-encoder rerank.

    Returns a list of (chunk, score) pairs sorted by descending rerank score.
    """
    scores = chunk_vectors @ query_vector
    order = np.argsort(-scores, kind="stable")[:retrieve_k]
    candidates = [chunks[int(index)] for index in order]
    rerank_scores = reranker.predict(
        [(query_text, item["text"]) for item in candidates], batch_size=batch_size
    )
    return sorted(zip(candidates, rerank_scores), key=lambda pair: -float(pair[1]))


def select_diverse_chunks(ranked, keep, max_chunks_per_page):
    """Greedily take the top-ranked chunks, capping how many share a page.

    Prevents overlapping windows of one section (e.g. the introduction) from
    crowding out a lower-ranked but distinct-page chunk. Ported unchanged
    from ask_pdf_question.py's select_diverse_chunks.
    """
    selected = []
    page_counts = {}
    for rank, (item, score) in enumerate(ranked, 1):
        if len(selected) >= keep:
            break
        page_key = (item["page_start"], item["page_end"])
        if page_counts.get(page_key, 0) >= max_chunks_per_page:
            continue
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        selected.append(dict(item, reranker_rank=rank, reranker_score=float(score)))
    return selected
