"""PDF text extraction, layout-preserving text, and page-image rendering.

extract_pages/render_page_images unify code that was duplicated verbatim in
run_general_pdf_review.py (inline in main()) and rerun_general_pdf_generation.py
(read_pdf/render_pages). The mechanics are identical in both places; only the
validation error wording differed between callers, so validation stays in the
CLI layer (see cli/retrieve.py and cli/generate.py) and this module never
raises on empty/text-free content — it just reports what it found.
"""
import statistics


def layout_text_from_pdfium(textpage):
    """Reconstruct a layout-preserving text rendering from a pdfium textpage.

    Ported unchanged from run_general_pdf_review.py.
    """
    characters = []
    widths = []
    for index in range(textpage.count_chars()):
        character = textpage.get_text_range(index, 1)
        if not character or character in "\r\n\x00":
            continue
        left, bottom, right, top = textpage.get_charbox(index)
        width = right - left
        height = top - bottom
        if width <= 0 or height <= 0:
            continue
        characters.append({
            "character": character,
            "left": left,
            "right": right,
            "center_y": (bottom + top) / 2,
            "height": height,
        })
        if not character.isspace():
            widths.append(width)
    if not characters:
        return ""

    median_width = statistics.median(widths) if widths else 5.0
    median_height = statistics.median(item["height"] for item in characters)
    y_tolerance = max(1.0, median_height * 0.35)
    lines = []
    for item in sorted(characters, key=lambda value: (-value["center_y"], value["left"])):
        line = next(
            (candidate for candidate in lines
             if abs(candidate["center_y"] - item["center_y"]) <= y_tolerance),
            None,
        )
        if line is None:
            line = {"center_y": item["center_y"], "characters": []}
            lines.append(line)
        line["characters"].append(item)

    output_lines = []
    for line in sorted(lines, key=lambda value: -value["center_y"]):
        parts = []
        previous_right = None
        for item in sorted(line["characters"], key=lambda value: value["left"]):
            if previous_right is not None and not item["character"].isspace():
                gap = item["left"] - previous_right
                if gap > median_width * 0.6:
                    parts.append(" " * min(40, max(1, round(gap / median_width))))
            parts.append(item["character"])
            previous_right = max(previous_right or item["right"], item["right"])
        output_lines.append("".join(parts).rstrip())
    return "\n".join(output_lines)


def select_layout_excerpt(text, context_lines=2, max_characters=12000):
    """Pull lines near evidence-relevant keywords (tables, metrics, limitations, ...).

    Ported unchanged from run_general_pdf_review.py.
    """
    keywords = (
        "table", "figure", "results", "dataset", "baseline", "accuracy",
        "precision", "recall", "f1", "error", "rmse", "auc", "ablation",
        "limitation", "validity", "%", "\u00b1",
    )
    lines = text.splitlines()
    selected_indexes = set()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            selected_indexes.update(range(start, end))
    excerpt = "\n".join(lines[index] for index in sorted(selected_indexes)).strip()
    return excerpt[:max_characters]


def extract_pages(pdf_path):
    """Extract per-page plain text and layout-preserved text from a PDF.

    Returns (pages, layout_page_texts) where pages is a list of
    {"page_number": int, "text": str}. Does not validate content (empty PDF,
    missing text layer) — see module docstring; callers validate with their
    own exact wording.
    """
    import pypdfium2 as pdfium

    pages = []
    layout_page_texts = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            text = textpage.get_text_range().strip()
            layout_page_texts.append(layout_text_from_pdfium(textpage))
            textpage.close()
            page.close()
            pages.append({"page_number": index + 1, "text": text})
    finally:
        document.close()
    return pages, layout_page_texts


def render_page_images(pdf_path, output_dir, page_count):
    """Render each page to a JPEG in output_dir; returns {page_number: path}.

    Ported unchanged (previously duplicated as run_general_pdf_review.py's
    inline loop and rerun_general_pdf_generation.py's render_pages).
    """
    import pypdfium2 as pdfium

    image_paths = {}
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for number in range(1, page_count + 1):
            page = document[number - 1]
            bitmap = page.render(scale=1.5)
            image = bitmap.to_pil().convert("RGB")
            path = (output_dir / f"page_{number:04d}.jpg").resolve()
            image.save(path, quality=90)
            image.close()
            bitmap.close()
            page.close()
            image_paths[number] = str(path)
    finally:
        document.close()
    return image_paths
