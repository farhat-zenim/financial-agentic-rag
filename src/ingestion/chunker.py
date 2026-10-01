"""
src/ingestion/chunker.py
────────────────────────
Stage 2 of the RAG ingestion pipeline.

Reads every *_clean.md from parsed_docs/markdown_clean/, enriches each chunk
with page numbers and section metadata from the matching Docling JSON in
parsed_docs/json/, then writes results to
parsed_docs/chunks/<doc_id>_chunks.jsonl  (one JSON object per line).

A running manifest is written to parsed_docs/chunk_manifest.json after every
document so a crash never loses progress.

Usage
-----
    python scripts/02_chunk_documents.py
    # or directly:
    python src/ingestion/chunker.py

Chunking strategy
-----------------
Pass 0 — Corruption repair
    A known upstream text corruption pattern (capital "S" replaced by "$" +
    adjacent digit, e.g. "NOTE $1.ignificant" → "NOTE 1. Significant") is
    repaired before any splitting occurs. A whitelist-based regex handles
    confirmed fragments; a generic detector flags remaining instances for QA.

Pass 1 — Heading-forward merge
    Splits the markdown on blank lines. Bare ## heading blocks are buffered
    and prepended to the NEXT content block instead of being emitted as
    standalone zero-information chunks (~17-25% of raw blocks).

Pass 2 — Type-aware splitting
    • Tables  : detected structurally (markdown separator row, e.g.
                "| --- | --- |"). Split on row boundaries if oversized, with
                the header + separator repeated on every part so each table
                chunk is independently interpretable.
    • Prose   : blocks over CHUNK_SIZE tokens are split by SentenceSplitter.
    Filters:
      - Drop chunks with word_count < MIN_WORDS (noise: "low", "None.")
      - Drop table chunks under Table-of-Contents section (page-number index)

Pass 3 — Small-chunk merge  (NEW)
    Consecutive prose chunks in the same section that are each under
    MERGE_THRESHOLD words are merged into one. This raises the average chunk
    from ~149 tokens to ~238 tokens, moving 56% of chunks into the 150-512
    token "ideal" range for bge-large-en-v1.5 (was 24%). Improves embedding
    density without splitting tables or crossing section boundaries.

text_for_embedding
    Every chunk — prose and table alike — gets a "Section: <title>\n\n"
    prefix so the embedding model has semantic context for continuation
    chunks and bare table rows. The prefix is applied AFTER merging so it
    isn't duplicated on merged chunks.

Output fields per chunk
-----------------------
    chunk_id            MD5(doc_id+index)[:12]  — stable for Qdrant upserts
    chunk_index         sequential position within the document
    doc_id              e.g. "3M_2015_10K"
    company             e.g. "3M"
    filing_type         "10K" | "10Q"
    fiscal_year         e.g. "2015"
    fiscal_quarter      "Q2" | null
    source              "SEC EDGAR"
    section             active ## heading at the time of the block
    page_no             from Docling JSON section_header provenance
    page_resolution     exact / normalized / prefix / fuzzy:<score> /
                        unresolved — QA signal for downstream filtering
    chunk_type          "prose" | "table"
    is_truncated        True if text_for_embedding still exceeds CHUNK_SIZE
                        tokens after all splitting (should be rare)
    possible_corruption True if text matches the generic corruption shape
                        after repair — QA signal, not acted on automatically
    word_count          word count of text (not text_for_embedding)
    char_count          character count of text
    text                original content (display / LLM context window)
    text_for_embedding  "Section: <title>\n\n<text>" — what the embedder sees

Prerequisites
-------------
    pip install llama-index-core tiktoken
"""

import difflib
import gc
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE    = Path("./parsed_docs")
CLEAN_MD_DIR   = OUTPUT_BASE / "markdown_clean"
JSON_DIR       = OUTPUT_BASE / "json"
CHUNKS_DIR     = OUTPUT_BASE / "chunks"
CHUNK_MANIFEST = OUTPUT_BASE / "chunk_manifest.json"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
CHUNK_SIZE               = 512    # token hard limit of bge-large-en-v1.5
CHUNK_OVERLAP            = 64     # token overlap for SentenceSplitter
MIN_WORDS                = 3      # drop chunks below this as noise
MERGE_THRESHOLD          = 300    # max words for small-prose merging (Pass 3)
FUZZY_PAGE_THRESHOLD     = 0.75   # difflib ratio floor for page resolution

# ── Token counting ────────────────────────────────────────────────────────────
# bge-large-en-v1.5 uses WordPiece; tiktoken's cl100k_base is a close proxy.
# Financial text over-segments vs plain English, so the fallback applies a
# 1.35x safety factor (calibrated on 3M tables: word-count alone under-flagged
# ~30% of oversized chunks as safe).
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    _SPLITTER_TOKENIZER = _ENC.encode
except Exception:
    _ENC = None
    logger.warning(
        "tiktoken unavailable — using word_count * 1.35 token estimate. "
        "Install tiktoken for accurate sizing."
    )
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.35)
    _SPLITTER_TOKENIZER = None

# ── Compiled regexes ──────────────────────────────────────────────────────────
_NOISE_RE = re.compile(
    r"(?i)^(table of contents|form 10-[kq]|united states|"
    r"securities and exchange|annual report|"
    r"for the (?:year|quarter)|part [ivx]+\s*$|"
    r"page \d+|quarterly period)"
)
_HEADING_RE     = re.compile(r"^##\s+(.+)$")
_TOC_SECTION_RE = re.compile(r"(?i)^(table of contents|index)$")
_TABLE_ROW_RE   = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE   = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# Corruption repair: "$<digits><fragment>" → "<digits> S<fragment>"
# Whitelist confirmed against real 3M filing text — do not add fragments
# without verifying they reconstruct a real word in context.
_CORRUPTION_FRAGMENTS = [
    "ignificant", "upplemental", "ettlement", "ecurities", "ecurity",
    "tock-Based", "tock", "ales", "elling", "hares", "enior", "elected",
    "tate", "ection", "ince", "ee", "cott", "imilar", "uch", "uccessors",
]
_CORRUPTION_RE = re.compile(
    r"\$(\d+)([.\s]{0,2})(" + "|".join(_CORRUPTION_FRAGMENTS) + r")\b",
    re.IGNORECASE,
)
# Detection-only: flags non-whitelisted instances of the same shape.
# Restricted to digit-letter runs (no space) to avoid false-positives on
# real dollar amounts like "$12 billion" which always have a space separator.
_GENERIC_CORRUPTION_RE = re.compile(r"\$\d+\.?[a-z]{2,}\b")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Alphanumeric-only, lowercase — used for fuzzy section matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _repair_known_corruption(text: str) -> tuple[str, int]:
    """Repair the whitelisted corruption pattern. Returns (text, n_repairs)."""
    count = 0
    def _sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        digits, sep, frag = m.group(1), m.group(2), m.group(3)
        sep_out = ". " if "." in sep else " "
        return f"{digits}{sep_out}S{frag}"
    return _CORRUPTION_RE.sub(_sub, text), count


def _is_markdown_table(block: str) -> bool:
    """
    Structural table detection: primary signal is a markdown separator row
    (e.g. "| --- | --- |"). Falls back to row-density for continuation blocks.
    More reliable than a pipe-count threshold for small tables.
    """
    lines = [l for l in block.split("\n") if l.strip()]
    if not lines:
        return False
    if any(_TABLE_SEP_RE.match(l) for l in lines):
        return True
    pipe_lines = sum(1 for l in lines if _TABLE_ROW_RE.match(l))
    return len(lines) >= 2 and pipe_lines / len(lines) >= 0.8


def _split_table(block: str, max_tokens: int) -> list[str]:
    """
    Split an oversized table on row boundaries with the header+separator
    repeated on every part, so each chunk is independently interpretable.
    Falls back to returning the whole block if no header is found.
    """
    lines     = block.split("\n")
    sep_idx   = next((i for i, l in enumerate(lines) if _TABLE_SEP_RE.match(l)), None)
    if sep_idx is None or sep_idx == 0:
        return [block]

    header_lines  = lines[:sep_idx + 1]
    header_text   = "\n".join(header_lines)
    header_tokens = count_tokens(header_text)
    rows          = [l for l in lines[sep_idx + 1:] if l.strip()]

    parts: list[str] = []
    current_rows:  list[str] = []
    current_tokens = header_tokens

    for row in rows:
        row_tokens = count_tokens(row)
        if current_rows and current_tokens + row_tokens > max_tokens:
            parts.append(header_text + "\n" + "\n".join(current_rows))
            current_rows   = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        parts.append(header_text + "\n" + "\n".join(current_rows))

    return parts or [block]


def _build_section_page_map(docling_json: dict) -> list[tuple[str, int]]:
    """Extract (section_title, page_no) pairs from Docling JSON."""
    section_pages: list[tuple[str, int]] = []
    for item in docling_json.get("texts", []):
        if item.get("label") != "section_header":
            continue
        title = item.get("text", "").strip()
        prov  = item.get("prov", [])
        page  = prov[0]["page_no"] if prov else 0
        if title and len(title) > 5 and not _NOISE_RE.match(title):
            section_pages.append((title, page))
    return section_pages


def _resolve_page(
    section_title: str,
    section_pages: list[tuple[str, int]],
) -> tuple[int, str]:
    """
    Map section_title to a page number. Returns (page_no, method).
    Priority: exact → normalized → prefix → fuzzy → unresolved(1).
    """
    if not section_title:
        return 1, "empty"

    for title, page in section_pages:
        if title.strip() == section_title.strip():
            return page, "exact"

    target_norm = _normalize(section_title)
    for title, page in section_pages:
        if _normalize(title) == target_norm:
            return page, "normalized"

    prefix = section_title[:15]
    for title, page in section_pages:
        if title.startswith(prefix) or section_title.startswith(title[:15]):
            return page, "prefix"

    best_page, best_score = 1, 0.0
    for title, page in section_pages:
        score = difflib.SequenceMatcher(
            None, target_norm, _normalize(title)
        ).ratio()
        if score > best_score:
            best_score, best_page = score, page
    if best_score >= FUZZY_PAGE_THRESHOLD:
        return best_page, f"fuzzy:{best_score:.2f}"

    return 1, "unresolved"


def _parse_doc_metadata(doc_id: str) -> dict:
    """
    Derive company / filing_type / period from filename convention:
      3M_2015_10K  |  ADOBE_2022Q2_10Q
    """
    parts          = doc_id.split("_")
    company        = parts[0] if parts else doc_id
    filing_type    = parts[-1] if len(parts) >= 2 else "UNKNOWN"
    period_raw     = parts[-2] if len(parts) >= 3 else ""
    q_match        = re.search(r"(\d{4})(Q\d)?", period_raw)
    fiscal_year    = q_match.group(1) if q_match else period_raw
    fiscal_quarter = q_match.group(2) if (q_match and q_match.group(2)) else None
    return {
        "doc_id":          doc_id,
        "company":         company,
        "filing_type":     filing_type,
        "fiscal_year":     fiscal_year,
        "fiscal_quarter":  fiscal_quarter,
        "source":          "SEC EDGAR",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3: small-chunk merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge_small_prose(chunks: list[dict]) -> list[dict]:
    """
    Merge consecutive prose chunks that share the same section title when
    their combined word count stays under MERGE_THRESHOLD.

    Why this matters for embedding quality
    ----------------------------------------
    bge-large-en-v1.5 produces its highest-quality vectors when input is
    150-350 tokens. Before this pass, 70-75% of chunks are under 150 words
    (~100 tokens), primarily because financial filings use many short
    paragraphs (a sentence or two per paragraph). Merging them within a
    section raises the average from ~149 to ~238 estimated tokens and moves
    56% of chunks into the 150-512 ideal range (was 24%), with zero new
    oversized chunks created.

    Guarantees
    ----------
    • Never crosses a section boundary
    • Never merges a prose chunk with a table chunk
    • Never produces a chunk over MERGE_THRESHOLD words
    • chunk_id is re-derived from the first constituent's chunk_index, so
      merged chunk IDs remain stable across re-runs (deterministic MD5)
    • chunk_index of the first constituent is preserved for ordering
    """
    if not chunks:
        return []

    merged: list[dict] = []
    i = 0

    while i < len(chunks):
        c = chunks[i]

        # Tables pass through unchanged
        if c["chunk_type"] != "prose":
            merged.append(c)
            i += 1
            continue

        # Absorb following prose chunks in the same section
        combined_text = c["text"]
        combined_wc   = c["word_count"]
        j = i + 1

        while j < len(chunks):
            nxt = chunks[j]
            if (
                nxt["chunk_type"] == "prose"
                and nxt["section"] == c["section"]
                and combined_wc + nxt["word_count"] <= MERGE_THRESHOLD
            ):
                combined_text += "\n\n" + nxt["text"]
                combined_wc   += nxt["word_count"]
                j += 1
            else:
                break

        if j > i + 1:
            # Build merged chunk from the first constituent's metadata
            mc                    = dict(c)
            mc["text"]            = combined_text
            mc["word_count"]      = combined_wc
            mc["char_count"]      = len(combined_text)
            # possible_corruption: True if ANY constituent was flagged
            mc["possible_corruption"] = any(
                chunks[k].get("possible_corruption", False)
                for k in range(i, j)
            )
            merged.append(mc)
        else:
            merged.append(c)

        i = j

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Core chunking function
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    doc_id:       str,
    md_text:      str,
    docling_json: dict,
) -> tuple[list[dict], dict]:
    """
    Split md_text into enriched chunk dicts ready for Qdrant ingestion.

    Returns (chunks, stats).
    """
    doc_meta      = _parse_doc_metadata(doc_id)
    section_pages = _build_section_page_map(docling_json)
    splitter      = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=_SPLITTER_TOKENIZER,
    )

    # ── Pass 0: corruption repair ─────────────────────────────────────────────
    md_text, corruption_repairs = _repair_known_corruption(md_text)
    if corruption_repairs:
        logger.info(
            "  Repaired %d corruption instance(s) in %s.",
            corruption_repairs, doc_id,
        )

    # ── Pass 1: heading-forward merge ─────────────────────────────────────────
    raw_blocks: list[str] = [b.strip() for b in md_text.split("\n\n") if b.strip()]
    merged_blocks: list[tuple[str, str]] = []
    pending_headings: list[str] = []
    current_section = "Preamble"
    dropped_heading = 0

    for block in raw_blocks:
        first_line = block.split("\n")[0].strip()
        m = _HEADING_RE.match(first_line)
        if m:
            current_section = m.group(1).strip()
            pending_headings.append(block)
            dropped_heading += 1
        else:
            if pending_headings:
                full_text = "\n\n".join(pending_headings) + "\n\n" + block
                pending_headings = []
            else:
                full_text = block
            merged_blocks.append((current_section, full_text.strip()))

    # Flush trailing headings that have no body text after them
    if pending_headings:
        merged_blocks.append((current_section, "\n\n".join(pending_headings)))

    # ── Pass 2: filter, type-aware splitting, per-chunk enrichment ────────────
    pre_merge_chunks: list[dict] = []
    chunk_index     = 0
    dropped_toc     = 0
    dropped_tiny    = 0
    unresolved_pages = 0

    for section, block in merged_blocks:
        is_table   = _is_markdown_table(block)
        chunk_type = "table" if is_table else "prose"
        page_no, page_method = _resolve_page(section, section_pages)
        if page_method == "unresolved":
            unresolved_pages += 1

        # Filter A: TOC tables — pure page-number index, no retrieval value
        if is_table and _TOC_SECTION_RE.match(section.strip()):
            dropped_toc += 1
            continue

        # Sub-split
        block_tokens = count_tokens(block)
        if block_tokens <= CHUNK_SIZE:
            sub_texts = [block]
        elif is_table:
            # Reserve 40 tokens for the "Section: <title>" prefix added below
            sub_texts = _split_table(block, CHUNK_SIZE - 40)
        else:
            sub_nodes = splitter.get_nodes_from_documents([Document(text=block)])
            sub_texts = [n.get_content() for n in sub_nodes] or [block]

        for text in sub_texts:
            wc = len(text.split())

            # Filter B: minimum word count
            if wc < MIN_WORDS:
                dropped_tiny += 1
                continue

            # QA flag — don't auto-repair, just surface
            possible_corruption = bool(_GENERIC_CORRUPTION_RE.search(text))

            pre_merge_chunks.append({
                "chunk_id":            hashlib.md5(
                                           f"{doc_id}_{chunk_index}".encode()
                                       ).hexdigest()[:12],
                "chunk_index":         chunk_index,
                "doc_id":              doc_meta["doc_id"],
                "company":             doc_meta["company"],
                "filing_type":         doc_meta["filing_type"],
                "fiscal_year":         doc_meta["fiscal_year"],
                "fiscal_quarter":      doc_meta["fiscal_quarter"],
                "source":              doc_meta["source"],
                "section":             section,
                "page_no":             page_no,
                "page_resolution":     page_method,
                "chunk_type":          chunk_type,
                "possible_corruption": possible_corruption,
                "word_count":          wc,
                "char_count":          len(text),
                "text":                text,
            })
            chunk_index += 1

    # ── Pass 3: small-prose merge ──────────────────────────────────────────────
    final_chunks_raw = _merge_small_prose(pre_merge_chunks)

    # ── Finalize: add text_for_embedding + is_truncated AFTER merging ─────────
    # Applied after merge so the prefix is never duplicated on merged chunks.
    chunks: list[dict] = []
    for c in final_chunks_raw:
        text_for_embedding = f"Section: {c['section']}\n\n{c['text']}"
        is_truncated       = count_tokens(text_for_embedding) > CHUNK_SIZE
        chunks.append({
            **c,
            "is_truncated":       is_truncated,
            "text_for_embedding": text_for_embedding,
        })

    stats = {
        "dropped_heading":     dropped_heading,
        "dropped_toc":         dropped_toc,
        "dropped_tiny":        dropped_tiny,
        "unresolved_pages":    unresolved_pages,
        "corruption_repairs":  corruption_repairs,
        "possible_corruption": sum(1 for c in chunks if c["possible_corruption"]),
        "merged_chunks":       len(pre_merge_chunks) - len(chunks),
    }
    return chunks, stats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_chunking_pipeline() -> None:
    clean_files = sorted(CLEAN_MD_DIR.glob("*_clean.md"))
    if not clean_files:
        logger.error(
            "No *_clean.md files found in '%s'. "
            "Run 01_ingest_documents.py first.",
            CLEAN_MD_DIR,
        )
        sys.exit(1)

    logger.info("Starting chunking pipeline — %d documents.", len(clean_files))

    manifest = {
        "timestamp":       datetime.now().isoformat(),
        "total_documents": len(clean_files),
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "min_words":       MIN_WORDS,
        "merge_threshold": MERGE_THRESHOLD,
        "documents":       [],
    }
    total_chunks = 0

    for idx, md_path in enumerate(clean_files, start=1):
        doc_id    = md_path.stem.replace("_clean", "")
        json_path = JSON_DIR / f"{doc_id}.json"

        logger.info("[%d/%d] Chunking: %s", idx, len(clean_files), doc_id)

        entry: dict = {
            "doc_id":              doc_id,
            "status":              "FAILED",
            "chunks_path":         None,
            "chunk_count":         0,
            "table_chunks":        0,
            "prose_chunks":        0,
            "truncated_chunks":    0,
            "merged_chunks":       0,
            "dropped_heading":     0,
            "dropped_toc":         0,
            "dropped_tiny":        0,
            "unresolved_pages":    0,
            "corruption_repairs":  0,
            "possible_corruption": 0,
            "error_log":           None,
        }

        try:
            md_text = md_path.read_text(encoding="utf-8")

            if not json_path.exists():
                logger.warning(
                    "  Docling JSON not found at '%s' — page numbers default to 1.",
                    json_path,
                )
                docling_json: dict = {}
            else:
                docling_json = json.loads(json_path.read_text(encoding="utf-8"))

            chunks, stats = chunk_document(doc_id, md_text, docling_json)

            out_path = CHUNKS_DIR / f"{doc_id}_chunks.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            table_n     = sum(1 for c in chunks if c["chunk_type"] == "table")
            prose_n     = sum(1 for c in chunks if c["chunk_type"] == "prose")
            truncated_n = sum(1 for c in chunks if c["is_truncated"])
            total_chunks += len(chunks)

            entry.update({
                "status":              "SUCCESS",
                "chunks_path":         str(out_path.resolve()),
                "chunk_count":         len(chunks),
                "table_chunks":        table_n,
                "prose_chunks":        prose_n,
                "truncated_chunks":    truncated_n,
                **stats,
            })

            logger.info(
                "  → %d chunks  (table=%d prose=%d truncated=%d merged=%d"
                "  dropped: heading=%d toc=%d tiny=%d"
                "  pages: unresolved=%d  corruption: repaired=%d flagged=%d)",
                len(chunks), table_n, prose_n, truncated_n,
                stats["merged_chunks"],
                stats["dropped_heading"], stats["dropped_toc"],
                stats["dropped_tiny"], stats["unresolved_pages"],
                stats["corruption_repairs"], stats["possible_corruption"],
            )

            if stats["unresolved_pages"] > 0:
                logger.warning(
                    "  ⚠ %d chunks have page_no=1 (unresolved). "
                    "Check sec_cleaner.py output for this document.",
                    stats["unresolved_pages"],
                )
            if stats["possible_corruption"] > 0:
                logger.warning(
                    "  ⚠ %d chunks flagged possible_corruption=True. "
                    "Inspect and extend _CORRUPTION_FRAGMENTS for this filer.",
                    stats["possible_corruption"],
                )

        except Exception as err:
            logger.error("  → FAILED: %s", err, exc_info=True)
            entry["error_log"] = str(err)

        finally:
            manifest["documents"].append(entry)
            CHUNK_MANIFEST.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            gc.collect()

    manifest["total_chunks"] = total_chunks
    CHUNK_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Chunking complete — %d total chunks across %d documents. "
        "Manifest: %s",
        total_chunks, len(clean_files), CHUNK_MANIFEST.resolve(),
    )


if __name__ == "__main__":
    run_chunking_pipeline()