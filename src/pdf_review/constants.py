"""Fixed vocabulary shared across the whole pipeline: topics, modes, categories.

Ported unchanged from run_general_pdf_review.py. No behavior change.
"""

TOPICS = {
    "purpose_contributions": {
        "subqueries": [
            "What research problem and practical motivation does the paper state?",
            "What contributions or novelty do the authors explicitly claim?",
        ],
    },
    "method": {
        "subqueries": [
            "What method, model, system, or procedure does the paper propose or evaluate?",
            "What inputs, data, assumptions, and implementation steps are explicitly described?",
        ],
    },
    "experiments_validation": {
        "subqueries": [
            "What datasets, baselines, experimental conditions, and evaluation metrics are used?",
            "What validation type is reported, such as simulation, benchmark, laboratory, field, or user study?",
        ],
    },
    "findings": {
        "subqueries": [
            "What are the paper's main quantitative findings?",
            "What qualitative findings or comparisons support the main conclusions?",
        ],
    },
    "limitations_review": {
        "subqueries": [
            "What limitations or threats to validity do the authors explicitly state?",
            "What important validation or evidence is limited according to explicit statements in the paper?",
        ],
    },
}

GENERATION_MODES = (
    "text_retrieval_text_only",
    "hybrid_retrieval_text_only",
    "hybrid_retrieval_with_images",
)

ALLOWED_CATEGORIES = {
    "purpose",
    "contribution",
    "method",
    "data",
    "assumption",
    "experiment",
    "validation",
    "finding",
    "limitation",
}

CATEGORY_SYNONYMS = {
    # -> finding
    "conclusion": "finding", "conclusions": "finding", "result": "finding",
    "results": "finding", "findings": "finding", "outcome": "finding",
    "outcomes": "finding", "observation": "finding", "observations": "finding",
    "comparison": "finding", "comparisons": "finding",
    # -> limitation
    "limitations": "limitation", "limits": "limitation", "weakness": "limitation",
    "weaknesses": "limitation", "drawback": "limitation", "drawbacks": "limitation",
    "constraint": "limitation", "constraints": "limitation", "caveat": "limitation",
    "caveats": "limitation", "threat": "limitation", "threats": "limitation",
    "shortcoming": "limitation", "shortcomings": "limitation",
    # -> contribution
    "contributions": "contribution", "novelty": "contribution",
    "innovation": "contribution", "innovations": "contribution",
    # -> method
    "methods": "method", "methodology": "method", "methodologies": "method",
    "procedure": "method", "procedures": "method", "algorithm": "method",
    "algorithms": "method", "technique": "method", "techniques": "method",
    "approach": "method", "approaches": "method", "model": "method",
    "models": "method", "system": "method", "framework": "method",
    "design": "method", "implementation": "method",
    # -> data
    "dataset": "data", "datasets": "data", "input": "data", "inputs": "data",
    # -> assumption
    "assumptions": "assumption", "premise": "assumption", "premises": "assumption",
    # -> experiment
    "experiments": "experiment", "setup": "experiment", "settings": "experiment",
    "configuration": "experiment", "scope": "experiment",
    # -> validation
    "validations": "validation", "evaluation": "validation",
    "evaluations": "validation", "benchmark": "validation",
    "benchmarks": "validation", "baseline": "validation", "baselines": "validation",
    "metric": "validation", "metrics": "validation", "indicator": "validation",
    "indicators": "validation", "criterion": "validation", "criteria": "validation",
    # -> purpose
    "purposes": "purpose", "goal": "purpose", "goals": "purpose",
    "objective": "purpose", "objectives": "purpose", "motivation": "purpose",
    "problem": "purpose", "rationale": "purpose", "background": "purpose",
}

OPTIONAL_EXTRACTION_FIELDS = ("method", "condition", "metric", "value", "unit")
