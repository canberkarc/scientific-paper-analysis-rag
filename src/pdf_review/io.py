"""JSON file I/O and VLM-response JSON repair.

Ported unchanged from run_general_pdf_review.py. No behavior change.
"""
import json
import re


def save(path, value):
    """Write JSON to a path that must not already exist (fails loudly on collision)."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def atomic_json(path, value):
    """Write JSON via a temp file + rename, so a crash mid-write never leaves a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_json_response(text):
    """Extract and repair a JSON object/array from a raw VLM text response.

    Tolerates a ```json fence, stray trailing punctuation, a trailing comma
    before a closing bracket, and multiple bare comma-separated numbers after
    a colon (wrapped into an array so the caller's validator can give a
    specific error instead of an opaque parse failure).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    object_start = cleaned.find("{")
    array_start = cleaned.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        raise ValueError("Model response does not contain a JSON object or array")
    start = min(starts)
    candidate = cleaned[start:]
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as error:
        repaired = candidate
        # Common VLM JSON mistakes, tried together as one repair pass:
        # (1) a trailing comma before a closing bracket, e.g. "...,}" or "...,]";
        # (2) multiple bare comma-separated numbers after a colon with no array
        #     brackets, e.g. "value": 0.78, 86.1, 0.61, (usually several metrics
        #     crammed into one field). Wrapping (2) into an array at least makes
        #     the JSON parseable, so the existing single-value validator can
        #     reject it with a specific, actionable error instead of an opaque
        #     parse failure.
        repaired = re.sub(
            r":(\s*)(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)(?=\s*[,}\]])",
            r":\1[\2]",
            repaired,
        )
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        try:
            value, end = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError:
            raise ValueError("Model response contains incomplete or invalid JSON") from error
        candidate = repaired
    trailing = candidate[end:].strip()
    if trailing.strip("`'\" \t\r\n"):
        raise ValueError("Model response contains text after the JSON value")
    return value
