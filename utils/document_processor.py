# utils/document_processor.py

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file) -> str:
    """
    Takes a PDF file object and extracts all text from it.
    Adds a [Page X] marker before each page so we know
    where in the document each chunk came from.
    """
    reader = PdfReader(file)
    full_text = ""

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # Some pages may be images with no extractable text
            full_text += f"\n\n[Page {page_num + 1}]\n{text}"

    return full_text


def load_txt(file) -> str:
    """
    Takes a .txt file object and reads it as a UTF-8 string.
    """
    return file.read().decode("utf-8")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits a long text into smaller overlapping chunks.

    Args:
        text:          The full document text
        chunk_size:    Max characters per chunk (500 chars ≈ ~125 tokens)
        chunk_overlap: How many characters overlap between adjacent chunks

    Returns:
        A list of text chunk strings ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Tries each separator in order before hard-cutting
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)
    return chunks