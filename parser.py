import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")
