import html
import re


def clean_sec_filing(raw_text: str) -> str:
    """
    Main orchestrator function to clean raw SEC filing parsed markdown text.
    Optimized for execution time on large (100k+ line) 10-K/10-Q documents.
    """
    # 0. Unescape HTML entities (&amp; -> &, &nbsp; -> space) and standardize line endings
    text = html.unescape(raw_text)
    text = text.replace('\r\n', '\n')

    text = _remove_metadata(text)
    text = _deduplicate_blocks(text)
    text = _correct_ocr_artifacts(text)
    text = _normalize_financial_data(text)
    text = _reconcile_sentence_flow(text)
    text = _normalize_tables(text)
    return text.strip()


# ---------------------------------------------------------------------------
# 1. Metadata removal
# ---------------------------------------------------------------------------

def _remove_metadata(text: str) -> str:
    # Clean dangling or incomplete HTML tags (e.g., `<a name=`)
    text = re.sub(r'<[^>]*?\bname=\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]*$', '', text)

    # Line-by-line filtering for repetitive headers/footers
    clean_lines = []
    for line in text.split('\n'):
        # Skip repetitive metadata headers/footers
        if re.search(
            r'(?i)(?:Form\s+10-[QK]|Quarterly Period Ended|3M COMPANY|ANNUAL REPORT ON FORM 10-K)',
            line,
        ):
            if "Table of Contents" in line or len(line.strip()) < 100:
                continue
        # Skip isolated "Table of Contents" or standalone page numbers
        line_str = line.strip()
        if line_str == "Table of Contents" or line_str.isdigit():
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


# ---------------------------------------------------------------------------
# 2. Block-level deduplication  (NEW — fixes Bug 5)
#    Removes repeated cover-page / boilerplate blocks that appear verbatim
#    more than once in the same filing (e.g. Activision 2016 repeats its
#    cover page 3x).  Only blocks of >=5 non-empty lines are deduplicated
#    so that short structural separators are not accidentally collapsed.
# ---------------------------------------------------------------------------

def _deduplicate_blocks(text: str, min_block_lines: int = 5) -> str:
    blocks = text.split('\n\n')
    seen: set = set()
    result = []
    for block in blocks:
        key = block.strip()
        if len([ln for ln in key.splitlines() if ln.strip()]) >= min_block_lines:
            if key in seen:
                continue
            seen.add(key)
        result.append(block)
    return '\n\n'.join(result)


# ---------------------------------------------------------------------------
# 3. OCR artifact correction  (Bug 1 fixed)
# ---------------------------------------------------------------------------

# 2-char suffix fragments that indicate a real OCR mid-word split.
# Deliberately excludes 'en' and 'in' to avoid false merges with prepositions
# like "factor in" -> "factorin".
_OCR_SUFFIX = r'(?:es|ed|er|ly|ue|re|ng|al|ic|ty|ry)'


def _correct_ocr_artifacts(text: str) -> str:
    # FIX (Bug 1): The original single regex fired on ANY word followed by a
    # single letter, merging normal English phrases such as "maintains a",
    # "filed a", "provides a" into "maintainsa", "fileda", "providesa".
    # Replaced with three targeted rules that only fire on genuine OCR patterns:

    # Rule A: apostrophe-s split — "Registrant' s" -> "Registrant's"
    text = re.sub(r"([A-Za-z]{3,})'\s+s\b", r"\1's", text)

    # Rule B: trailing-s split on section headings — "Factor s" -> "Factors",
    # "Comment s" -> "Comments", "Proceeding s" -> "Proceedings".
    # Guard: first token must end in a non-vowel, non-s consonant so that
    # complete words like "provides" or "versus" are not incorrectly extended.
    text = re.sub(r'\b([A-Za-z]{4,}[^aeiouAEIOUsS\W])\s+s\b', r'\1s', text)

    # Rule B2: explicit fix for "Busines s" -> "Business" (first token ends in
    # 'es' so Rule B's consonant guard above cannot catch it).
    text = re.sub(r'\bBusines\s+s\b', 'Business', text, flags=re.IGNORECASE)

    # Rule C: 2-char morpheme splits — "Reven ue" -> "Revenue",
    # "manufactur ed" -> "manufactured".
    text = re.sub(
        rf'\b([A-Za-z]{{4,}})\s+({_OCR_SUFFIX})(?=[\s\.,;:\?\!]|$)',
        r'\1\2',
        text,
    )

    # Map Cyrillic ПЕМ to standard ITEM format
    text = re.sub(r'ПЕМ\s*(\d+)', r'ITEM \1', text)
    text = text.replace('Π', '')

    # Anchor detached currency symbols
    text = re.sub(r'(?:S|\\\$)\s*(\(?\d[\d,\.]*\)?)', r'$\1', text)
    text = re.sub(r'(\(?\d[\d,\.]*\)?)\s*(?:S|\\\$)', r'$\1', text)
    return text


# ---------------------------------------------------------------------------
# 4. Financial-data normalisation  (Bugs 2 & 3 fixed)
# ---------------------------------------------------------------------------

def _normalize_financial_data(text: str) -> str:
    # FIX (Bug 2): The original paren-to-negative regex had no context guard,
    # so it corrupted phone numbers in every filing.
    # e.g. "(651) 733-1110" became "-651 733-1110", and the currency-symbol
    # regex then produced "-651 733-$1110...".
    # New pattern excludes:
    #   • sequences followed by space+digits (phone exchange guard)
    #   • sequences followed by a dash or dot then digit (phone tail guard)
    text = re.sub(
        r'(?<!\w)\(\s*(\$?\d[\d,\.]*)\s*\)(?!\s*[\-\.\d])',
        r'-\1',
        text,
    )

    # FIX (Bug 3): The thousand-separator fix incorrectly converted
    # "$1.763 billion" -> "$1,763 billion" — a fabricated number.
    # New pattern excludes matches followed by a monetary-scale word.
    text = re.sub(
        r'(\b\d{1,3})\.(\d{3}\b)(?!\s*(?:billion|million|trillion))',
        r'\1,\2',
        text,
    )
    return text


# ---------------------------------------------------------------------------
# 5. Sentence-flow reconciliation
# ---------------------------------------------------------------------------

def _reconcile_sentence_flow(text: str) -> str:
    # Rejoin hyphenated words split across line wraps
    text = re.sub(r'([a-zA-Z]+)-\n\s*([a-zA-Z]+)', r'\1\2', text)
    # Collapse redundant blank lines
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text


# ---------------------------------------------------------------------------
# 6. Table normalisation  (Bug 4 fixed)
# ---------------------------------------------------------------------------

def _normalize_tables(text: str) -> str:
    # FIX (Bug 4): The original function reformatted spacing inside pipe lines
    # but never removed structural rows with no real content (<=1 non-empty
    # cell).  These pure-structure rows — TOC dividers such as
    # "| |  | Beginning Page |" and "| PARTI |  | |" — made up virtually the
    # entire "malformed table" count and carried no analytical value.
    lines = text.split('\n')
    normalized = []
    for line in lines:
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            non_empty = [p for p in parts if p]
            # Drop rows that are pure structure with no real data
            if len(non_empty) <= 1:
                continue
            line = '| ' + ' | '.join(non_empty) + ' |'
        normalized.append(line)
    return '\n'.join(normalized)