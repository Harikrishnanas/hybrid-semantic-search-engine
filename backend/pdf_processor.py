import os
import re
import logging
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF text extraction, cleaning, and intelligent semantic chunking.
    Uses section-aware splitting and automatic chunk-size determination.
    """

    # ------------------------------------------------------------------ #
    # Chunk-size bounds                                                    #
    # ------------------------------------------------------------------ #
    _MIN_CHUNK_CHARS = 800
    _MAX_CHUNK_CHARS = 2500
    _TARGET_WORDS = 300
    _OVERLAP_SENTENCES = 2      # Overlap two sentences between chunks for context

    # ------------------------------------------------------------------ #
    # Section header pattern (all-caps lines, or lines ending with colon) #
    # ------------------------------------------------------------------ #
    _SECTION_HEADER_RE = re.compile(
        r"^(?:[A-Z][A-Z &/,\-]{2,}|.{3,60}:)\s*$", re.MULTILINE
    )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_text_and_pages(file_path: str) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Extracts text from a PDF page-by-page, recording start/end char
        indices for each page so we can map chunks back to page numbers.

        Returns:
            (merged_text, page_ranges)
            where page_ranges is a list of (start_char, end_char, page_number)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        merged_text = ""
        page_ranges: List[Tuple[int, int, int]] = []

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logger.info(f"Extracting text from {file_path} ({total_pages} pages)")

            for index, page in enumerate(reader.pages):
                page_num = index + 1
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    page_text = ""


                # Light cleaning that preserves structure
                page_text_clean = PDFProcessor._clean_page_text(page_text)

                if page_text_clean:
                    separator = "\n\n" if merged_text else ""
                    start_char = len(merged_text) + len(separator)
                    merged_text += separator + page_text_clean
                    end_char = len(merged_text)
                    page_ranges.append((start_char, end_char, page_num))

            logger.info(
                f"Extracted {len(merged_text)} characters from "
                f"{len(page_ranges)} non-empty pages."
            )
            return merged_text, page_ranges

        except Exception as exc:
            logger.error(f"Error extracting PDF: {exc}")
            raise

    @staticmethod
    def get_page_number(char_pos: int, page_ranges: List[Tuple[int, int, int]]) -> int:
        """Maps a character position to the correct PDF page number."""
        if not page_ranges:
            return 1
        for start, end, page_num in page_ranges:
            if start <= char_pos <= end:
                return page_num
        for start, end, page_num in page_ranges:
            if char_pos <= end:
                return page_num
        return page_ranges[-1][2]

    @classmethod
    def chunk_pdf(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF and produces focused, semantically coherent
        text chunks with page-number metadata.

        The algorithm:
        1. Extract and clean text.
        2. Split into *sections* using header detection.
        3. Within each section, split into sentences.
        4. Pack sentences into chunks respecting auto-computed size limits.
        5. Apply 1-sentence overlap between adjacent chunks.

        Returns:
            [{"text": "...", "page_number": N, "chunk_index": M}, ...]
        """
        merged_text, page_ranges = cls.extract_text_and_pages(file_path)

        if not merged_text.strip():
            logger.warning("No text extracted. Document may be empty or image-only.")
            return []

        # 1. Compute target chunk size from document
        target_size = cls._auto_chunk_size(merged_text)
        logger.info(f"Auto target chunk size → {target_size} chars")

        # 2. Split into semantic sections, then into sentences
        sections = cls._split_into_sections(merged_text)
        logger.info(f"Detected {len(sections)} sections in document.")

        # 3. Pack into chunks
        all_chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        for section_text, section_start_char in sections:
            sentences = cls._split_into_sentences(section_text)
            if not sentences:
                continue
            print("\n" + "=" * 80)
            print("SECTION")
            print(section_text[:1000])
            print("=" * 80)    
            packed = cls._pack_sentences(
                sentences, target_size, section_start_char, page_ranges
            )
            for chunk_text, page_num in packed:
                print("\nCHUNK:")
                print(chunk_text)
                print("-" * 60)
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        logger.info(f"Generated {len(all_chunks)} chunks from PDF.")
        return all_chunks

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_page_text(raw: str) -> str:
        """Cleans raw page text while preserving structural newlines."""
        # Merge hyphenated line-breaks
        text = re.sub(r"-\n(\S)", r"\1", raw)
        # Collapse multiple spaces/tabs within a line
        text = re.sub(r"[ \t]+", " ", text)
        # Normalise paragraph separators
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip per-line whitespace
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()

    @classmethod
    def _auto_chunk_size(cls, text: str) -> int:
        """
        Adaptive chunk size for general PDFs.
        Produces much larger, meaningful chunks.
        """

        words = len(text.split())

        if words < 500:
            return 1000

        elif words < 3000:
            return 1500

        elif words < 10000:
            return 2000

        return 2500

    @classmethod
    def _split_into_sections(cls, text: str) -> List[Tuple[str, int]]:
        """
        Splits text at section headers (ALL-CAPS lines, colon-ending headers).
        Returns list of (section_text, start_char_in_merged_text).
        If no headers are detected, returns the whole text as one section.
        """
        # Find all header positions
        headers = list(cls._SECTION_HEADER_RE.finditer(text))

        if not headers:
            # No headers found — return entire text as a single section
            return [(text, 0)]

        sections: List[Tuple[str, int]] = []

        # Content before the first header (if any)
        if headers[0].start() > 0:
            pre = text[: headers[0].start()].strip()
            if pre:
                sections.append((pre, 0))

        # Each header + body until next header
        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_text, start))

        return sections

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into meaningful sentences while preserving context.
        """

        text = re.sub(r"[\u2022\uf0b7]", ". ", text)

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > 10
        ]
    @classmethod
    def _pack_sentences(
        cls,
        sentences: List[str],
        target_size: int,
        section_start_char: int,
        page_ranges: List[Tuple[int, int, int]],
    ) -> List[Tuple[str, int]]:
        """
        Creates larger semantic chunks using sentence packing.
        """

        results = []

        current = []
        current_len = 0

        for sentence in sentences:

            sentence_len = len(sentence)

            if current and current_len + sentence_len > target_size:

                chunk_text = " ".join(current)

                page_num = PDFProcessor.get_page_number(
                    section_start_char,
                page_ranges
            )

                results.append(
                    (chunk_text, page_num)
                )

                overlap = current[-2:] if len(current) >= 2 else current

                current = overlap.copy()

                current_len = len(
                    " ".join(current)
                )

            current.append(sentence)

            current_len += sentence_len

        if current:

                chunk_text = " ".join(current)

                page_num = PDFProcessor.get_page_number(
                    section_start_char,
                    page_ranges
            )

                results.append(
                    (chunk_text, page_num)
                )

        return results
