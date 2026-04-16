import re
import fitz
import pdfplumber
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def is_garbled(text):
    if not text or len(text.strip()) < 30:
        return True
    if text.count('/G') > 10:
        return True
    if text.count('\ufffd') > 5:
        return True
    if text.count('(cid:') > 5:
        return True
    chars = [c for c in text if c != '\n' and c != ' ']
    if not chars:
        return True
    printable_ascii = sum(1 for c in chars if 32 <= ord(c) < 127)
    ratio = printable_ascii / len(chars)
    if ratio < 0.6:
        return True
    words = text.lower().split()
    if len(words) < 10:
        return True
    common_words = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and',
                    'or', 'in', 'for', 'on', 'at', 'by', 'be', 'not',
                    'if', 'it', 'do', 'no', 'up', 'as', 'so', 'can',
                    'this', 'that', 'with', 'from', 'all', 'any', 'has'}
    found = sum(1 for w in words if w in common_words)
    if found < 1:
        return True
    if is_menu_diagram(text):
        return True
    return False


def is_menu_diagram(text):
    """Detect pages that are menu navigation diagrams — not useful content."""
    if not text:
        return False
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    arrow_lines = sum(1 for l in lines if l in ['↑', '↓', '→', '←', 'Menu', 'Main'])
    short_lines = sum(1 for l in lines if len(l) < 20)
    arrow_ratio = arrow_lines / len(lines)
    short_ratio = short_lines / len(lines)
    if arrow_ratio > 0.15 and short_ratio > 0.7:
        return True
    return False


def extract_page_pymupdf(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text()
    doc.close()
    return text.strip()


def extract_page_pdfplumber(pdf_path, page_num):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text()
        return text.strip() if text else ""


def extract_page_ocr(pdf_path, page_num):
    images = convert_from_path(
        pdf_path,
        first_page=page_num + 1,
        last_page=page_num + 1,
        dpi=300
    )
    if not images:
        return ""
    text = pytesseract.image_to_string(images[0], lang='eng')
    return text.strip()


def load_pdf_robust(pdf_path, filename):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    documents = []
    stats = {"pymupdf": 0, "pdfplumber": 0, "ocr": 0, "skipped": 0}

    for page_num in range(total_pages):
        text = ""
        strategy = ""

        # Strategy 1: PyMuPDF
        try:
            text = extract_page_pymupdf(pdf_path, page_num)
            if not is_garbled(text):
                strategy = "pymupdf"
        except Exception:
            text = ""

        # Strategy 2: pdfplumber
        if not strategy:
            try:
                text = extract_page_pdfplumber(pdf_path, page_num)
                if not is_garbled(text):
                    strategy = "pdfplumber"
            except Exception:
                text = ""

        # Strategy 3: OCR
        if not strategy:
            try:
                print(f"  Page {page_num + 1}: running OCR...")
                text = extract_page_ocr(pdf_path, page_num)
                if text and len(text.strip()) > 30:
                    strategy = "ocr"
            except Exception as e:
                print(f"  Page {page_num + 1}: OCR failed - {e}")

        if strategy:
            stats[strategy] += 1
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "extraction_strategy": strategy
                }
            ))
        else:
            stats["skipped"] += 1
            print(f"  Page {page_num + 1}: skipped - no readable text")

    print(f"  Extraction summary: {stats}")
    return documents


# ---------------------------------------------------------------------------
# Semantic chunking helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _split_into_sentences(text):
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _find_split_indices(similarities, percentile=25):
    """
    Return indices where a split should occur.
    Splits where similarity drops below the given percentile threshold.
    Lower percentile = fewer splits (larger chunks).
    Higher percentile = more splits (smaller chunks).
    """
    threshold = np.percentile(similarities, percentile)
    return [i for i, sim in enumerate(similarities) if sim < threshold]


def load_pdf_semantic(pdf_path, filename, percentile=25):
    """
    Load a PDF and chunk it semantically using embedding-based sentence similarity.

    Strategy:
    1. Extract pages using load_pdf_robust() — reuses the full extraction cascade
    2. Split each page into sentences
    3. Embed all sentences in one batched API call
    4. Split where cosine similarity between adjacent sentences drops below threshold
    5. Return LangChain Document objects with source metadata

    Args:
        pdf_path:   Path to the PDF file
        filename:   Original filename (stored in metadata)
        percentile: Similarity percentile below which a split is made (default 25)
                    Lower = bigger chunks, Higher = smaller chunks

    Returns:
        List of Document objects with semantic chunk boundaries
    """
    load_dotenv(Path(__file__).parent.parent / '.env', override=True)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 1: reuse robust PDF extraction
    print("Extracting pages with load_pdf_robust()...")
    pages = load_pdf_robust(pdf_path, filename)
    print(f"  {len(pages)} pages extracted")

    # Step 2: split all pages into sentences, preserving page metadata
    all_sentences = []
    all_metadata = []

    for doc in pages:
        sentences = _split_into_sentences(doc.page_content)
        for sentence in sentences:
            all_sentences.append(sentence)
            all_metadata.append(doc.metadata.copy())

    print(f"  {len(all_sentences)} sentences extracted across all pages")

    if not all_sentences:
        print("  Warning: no sentences extracted")
        return []

    # Step 3: embed all sentences in one batched API call
    print("Embedding sentences (single batched API call)...")
    embeddings = embeddings_model.embed_documents(all_sentences)
    print(f"  {len(embeddings)} sentence embeddings computed")

    # Step 4: compute cosine similarity between adjacent sentences
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Step 5: find split points
    split_indices = set(_find_split_indices(similarities, percentile=percentile))
    print(f"  {len(split_indices)} split points identified (percentile={percentile})")

    # Step 6: group sentences into chunks at split boundaries
    chunks = []
    current_sentences = [all_sentences[0]]
    current_metadata = all_metadata[0]

    for i in range(1, len(all_sentences)):
        if (i - 1) in split_indices:
            chunks.append(Document(
                page_content=" ".join(current_sentences),
                metadata={
                    **current_metadata,
                    "chunking_strategy": "semantic",
                    "chunk_index": len(chunks)
                }
            ))
            current_sentences = [all_sentences[i]]
            current_metadata = all_metadata[i]
        else:
            current_sentences.append(all_sentences[i])

    # Commit final chunk
    if current_sentences:
        chunks.append(Document(
            page_content=" ".join(current_sentences),
            metadata={
                **current_metadata,
                "chunking_strategy": "semantic",
                "chunk_index": len(chunks)
            }
        ))

    print(f"  {len(chunks)} semantic chunks created")
    return chunks