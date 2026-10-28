"""Semantic section detection and token-window chunking.

Ported unchanged from run_general_pdf_review.py. build_chunks() factors out
the previously inline chunk-building loop from main() as a pure function
(same logic, same output shape); it does not write sections.json itself —
the CLI does that, exactly as before.
"""
import re

TOP_LEVEL_HEADING = re.compile(
    r"^(?:[IVXLCDM]+\.|\d+(?:\.\d+)*\.?)\s+[A-Z][A-Za-z0-9()/:,&.'\- ]{1,120}$"
)
SUBSECTION_HEADING = re.compile(
    r"^[A-Z]\.(?:\d+\.)?\s+[A-Z][A-Za-z0-9()/:,&.'\- ]{1,120}$"
)
NAMED_HEADING = re.compile(
    r"^(?:ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHOD|METHODS|METHODOLOGY|"
    r"MATERIALS AND METHODS|EXPERIMENTS?|EXPERIMENTAL SETUP|RESULTS?|DISCUSSION|"
    r"LIMITATIONS?|CONCLUSION|CONCLUSIONS|REFERENCES|BIBLIOGRAPHY|"
    r"ACKNOWLEDGMENTS?|APPENDIX(?:\s+[A-Z])?)$",
    re.IGNORECASE,
)


def windows(tokens, size, overlap):
    for start in range(0, len(tokens), size - overlap):
        end = min(start + size, len(tokens))
        yield start, end, tokens[start:end]
        if end == len(tokens):
            break


def is_heading(line):
    candidate = " ".join(line.replace("\ufffe", "").replace("\u00ad", "").split())
    if not candidate or len(candidate) > 140:
        return False
    if candidate.lower().startswith(("fig.", "figure ", "table ")):
        return False
    if re.match(r"^\d+\s+(?:Department|University|School|Institute)\b", candidate, re.IGNORECASE):
        return False
    return bool(
        TOP_LEVEL_HEADING.fullmatch(candidate)
        or SUBSECTION_HEADING.fullmatch(candidate)
        or NAMED_HEADING.fullmatch(candidate)
    )


def make_semantic_sections(pages):
    sections = []
    current = None

    def start_section(title, page_number):
        return {
            "section_id": len(sections),
            "title": title,
            "page_start": page_number,
            "page_end": page_number,
            "lines": [],
            "page_fragments": {},
        }

    def finish_section(section):
        if section is None:
            return
        text = "\n".join(section.pop("lines")).strip()
        if text:
            section["page_fragments"] = [
                {"page_number": page_number, "text": "\n".join(lines).strip()}
                for page_number, lines in sorted(section["page_fragments"].items())
                if "\n".join(lines).strip()
            ]
            section["text"] = text
            sections.append(section)

    def append_line(section, page_number, line):
        section["lines"].append(line)
        section["page_fragments"].setdefault(page_number, []).append(line)

    for page in pages:
        page_number = page["page_number"]
        for raw_line in page["text"].splitlines():
            line = " ".join(raw_line.replace("\ufffe", "").replace("\u00ad", "").split())
            if not line:
                continue
            if is_heading(line):
                finish_section(current)
                current = start_section(line, page_number)
                append_line(current, page_number, line)
            else:
                if current is None:
                    current = start_section("Document preamble", page_number)
                current["page_end"] = page_number
                append_line(current, page_number, line)
        if current is not None:
            current["page_end"] = page_number
    finish_section(current)

    for section_id, section in enumerate(sections):
        section["section_id"] = section_id
    return sections


def build_chunks(pages, tokenizer, segmentation, chunk_tokens, overlap_tokens):
    """Build the chunk list for either segmentation mode.

    Returns (chunks, sections); sections is [] for segmentation="page".
    Pure function — the caller is responsible for persisting sections.json.
    """
    chunks = []
    if segmentation == "semantic":
        sections = make_semantic_sections(pages)
        for section in sections:
            for fragment in section["page_fragments"]:
                page_number = fragment["page_number"]
                fragment_text = fragment["text"]
                if not fragment_text.startswith(section["title"]):
                    fragment_text = f"{section['title']}\n{fragment_text}"
                tokens = tokenizer.encode(fragment_text, add_special_tokens=False,
                                          truncation=False)
                for start, end, ids in windows(tokens, chunk_tokens, overlap_tokens):
                    chunks.append({
                        "chunk_id": f"s{section['section_id']}_p{page_number}_t{start}",
                        "section_id": section["section_id"],
                        "section_title": section["title"],
                        "page_number": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                        "token_start": start,
                        "token_end": end,
                        "text": tokenizer.decode(ids, skip_special_tokens=True),
                    })
    else:
        sections = []
        for page in pages:
            tokens = tokenizer.encode(page["text"], add_special_tokens=False, truncation=False)
            for start, end, ids in windows(tokens, chunk_tokens, overlap_tokens):
                chunks.append({
                    "chunk_id": f"p{page['page_number']}_t{start}",
                    "page_number": page["page_number"],
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                    "token_start": start,
                    "token_end": end,
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                })
    return chunks, sections
