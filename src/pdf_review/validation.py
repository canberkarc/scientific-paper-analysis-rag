"""Strict, shared validators for extraction rows and reused evidence.

Ported unchanged from run_general_pdf_review.py. No behavior change.
"""
from .constants import OPTIONAL_EXTRACTION_FIELDS
from .normalization import INVALID_NULL_STRINGS, NUMERIC_VALUE, normalize_field_list, normalize_range_value


def validate_optional_extraction_fields(row):
    if "value" in row:
        row["value"] = normalize_range_value(row["value"])
    for field in OPTIONAL_EXTRACTION_FIELDS:
        if field != "value" and field in row:
            row[field] = normalize_field_list(row[field])
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a string, number, or JSON null")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                row[field] = None
                continue
            if text.casefold() in INVALID_NULL_STRINGS:
                row[field] = None
                continue
        elif field != "value" or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a string, number, or JSON null")

    value = row.get("value")
    unit = row.get("unit")
    if value is None:
        if unit is not None:
            row["unit"] = None
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return
    if not NUMERIC_VALUE.fullmatch(str(value).strip()):
        raise ValueError(
            "value must contain only a numerical value without its unit; "
            "put qualitative information in claim and use JSON null for value"
        )


def validate_reused_evidence(evidence, selected_topics, page_count):
    if not isinstance(evidence, dict):
        raise ValueError("Reused evidence must be a JSON object")
    if set(evidence) != set(selected_topics):
        raise ValueError(
            "Reused evidence topics do not match --topics: "
            f"expected {sorted(selected_topics)}, found {sorted(evidence)}"
        )
    for topic_name, topic in selected_topics.items():
        traces = evidence[topic_name].get("subquery_traces")
        if not isinstance(traces, list) or len(traces) != len(topic["subqueries"]):
            raise ValueError(f"Invalid reused evidence traces for topic {topic_name}")
        for expected_query, trace in zip(topic["subqueries"], traces):
            if trace.get("subquery") != expected_query:
                raise ValueError(
                    f"Reused evidence query mismatch for topic {topic_name}: "
                    f"expected {expected_query!r}, found {trace.get('subquery')!r}"
                )
            if not isinstance(trace.get("selected_text"), list):
                raise ValueError(f"Missing selected_text for query {expected_query!r}")
            for key in ("selected_text_pages", "selected_image_pages"):
                page_numbers = trace.get(key)
                if not isinstance(page_numbers, list):
                    raise ValueError(f"Missing {key} for query {expected_query!r}")
                if any(
                    not isinstance(page, int) or page < 1 or page > page_count
                    for page in page_numbers
                ):
                    raise ValueError(f"Invalid page in {key} for query {expected_query!r}")
