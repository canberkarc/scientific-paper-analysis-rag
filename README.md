# pdf-review

Evidence-grounded, multimodal RAG review pipeline for scientific PDFs.
BGE-M3 retrieval + reranking, Qwen2.5-VL visual page scoring and structured
extraction, an evidence-ref-based reviewer, and a structural audit — with an
automated readable analysis report and an optional manual semantic-audit
worksheet.

## What it does

Given a PDF, the pipeline:

1. Extracts text and detects section headings.
2. Chunks the text and retrieves the most relevant chunks per topic with
   BGE-M3 + a cross-encoder reranker, capping how many chunks may come from
   the same page (page-diversity selection).
3. Scores whole page images for visual relevance with Qwen2.5-VL and selects
   the most useful pages per topic.
4. Extracts structured, evidence-cited findings (Qwen2.5-VL) across five
   fixed topics: purpose/contributions, method, experiments/validation,
   findings, and limitations — each in three modes (text-only retrieval,
   hybrid text, hybrid text+images) so text/context/pixel contribution can
   be compared.
5. Runs an evidence-grounded reviewer that cites exact extraction records
   (`evidence_ref`), never treats a coverage gap as a stated defect of the
   paper unless a cited record is explicitly a limitation, and puts genuine
   uncertainty into unresolved checks instead of forcing a recommendation.
6. Runs a structural audit: every citation must reference a retrieved page,
   flags numeric values not found in the extracted page text, and flags a
   likely bracketed-citation-number-as-value confusion (e.g. copying `[19]`
   as `value=19`) as a manual-review signal, never an automatic rejection.
7. Produces an automated, readable analysis report (`analysis_report.md`)
   directly from the audited output, clearly marked "not manually verified."
8. Optionally builds a manual semantic-audit worksheet per generation mode,
   and/or answers one ad-hoc question against the same cached retrieval
   index and strict schema.

Semantic correctness is never assumed from structural validity alone —
`automatic_semantic_claim_verification` is always `false` in `audit.json`.
A human should verify any specific claim, number, or page citation before
relying on it.

## Structure

```
src/pdf_review/
    constants.py        topics, generation modes, category enum/synonyms
    io.py                JSON I/O, VLM-response JSON repair
    normalization.py     recovers near-miss VLM output shapes
    validation.py        strict field/evidence validators
    pdf.py                text/layout extraction, page-image rendering
    segmentation.py      heading detection, semantic sections, chunking
    retrieval.py          dense retrieval, reranking, page-diversity selection
    visual_retrieval.py  whole-page visual relevance scoring
    vlm.py                Qwen2.5-VL loading + generate/generate_json
    extraction.py        strict structured-finding extraction
    reviewer.py           evidence-grounded reviewer
    audit.py              citation checks, structural audit, condition comparison
    question.py           ad-hoc question answering
    reporting/
        analysis_report.py   automated readable report
        claim_worksheet.py   manual semantic-audit worksheet
    cli/                  thin argparse wrappers; import core, never the reverse
        retrieve.py        retrieval + visual page scoring -> evidence.json
        generate.py        strict extraction, reviewer, audit from evidence.json
        ask.py             ad-hoc question against cached evidence
        render_report.py  builds analysis_report.md
        build_worksheet.py builds the manual audit worksheets
        summarize.py       prints a run's audit/condition-comparison summary

scripts/slurm/           .sbatch job files + common_env.sh
tests/
    unit/                 CPU-only, seconds; each case is a real historical
                         VLM failure, not a hypothetical
    integration/          CPU-only, fixture-based (reporting pipeline)
    fixtures/
runs/                     job output goes here (gitignored)
```

## Output layout (per run)

```
runs/<name>/
    evidence.json, chunks.json, pages.json, sections.json,
    document_stats.json, page_*.jpg          # from `retrieve`
    intermediate_extractions.json,
    reviewer_assessment.json, review_results.json,
    audit.json, condition_comparison.json,
    analysis_report.md, audit_worksheets/    # from `generate` + reporting
    answer.json                              # from `ask`, if used
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`scripts/slurm/common_env.sh` also sets `PYTHONPATH` so
`python -m pdf_review.cli.*` works even without installing the package, but
`pip install -e .` in the cluster venv is still the recommended path.

## Usage

Full pipeline (new PDF):
```bash
sbatch scripts/slurm/run_pipeline.sbatch /path/to/paper.pdf runs/paper1 ["optional ad-hoc question"]
```

Manual two-step (reusing evidence):
```bash
sbatch scripts/slurm/build_evidence_v8.sbatch paper.pdf runs/paper1/evidence
sbatch scripts/slurm/rerun_general_pdf_generation_a100.sbatch paper.pdf runs/paper1/evidence/evidence.json runs/paper1/strict
```

Ad-hoc question against cached evidence:
```bash
sbatch scripts/slurm/ask_pdf_question.sbatch runs/paper1/evidence runs/paper1/question "What forgetting factor is used?"
```

Print a run's summary:
```bash
python -m pdf_review.cli.summarize --strict-dir runs/paper1/strict
```

## Tests

```bash
pytest tests/
```

No GPU needed — every test exercises pure Python logic (normalization,
validation, extraction, reviewer, audit, retrieval selection, reporting)
against fixtures, not live model calls.

## Known limitations

- `automatic_semantic_claim_verification` is always `false`: the pipeline
  checks that a citation points to a retrieved page and that a numeric value
  appears in the extracted page text, but never confirms a claim actually
  means what the paper says. Manual review (`build_worksheet`) is required
  for anything load-bearing.
- A numeric value visible only in page pixels (not in the extracted text
  layer) will not be found by the citation check even when correct.
- The same Qwen2.5-VL model performs both extraction and reviewer
  generation, so a systematic misreading can appear in both stages.
- Scanned PDFs without an extractable text layer are not supported.
