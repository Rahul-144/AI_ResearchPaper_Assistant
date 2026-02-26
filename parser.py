
from langchain_community.document_loaders import PyMuPDFLoader

# Open the PDF file
def load_pdf(file_path):
    # Create a PDF loader object
    loader = PyMuPDFLoader(file_path)
    
    # Load the PDF file
    document = loader.load()
    
    print (f"Loaded {len(document)} pages from {file_path}")
    
    # Initialize an empty list to store the text content
    text_content = []
    
    # Iterate over each page and extract the text content
    for page in document:
        text_content.append(page.page_content)
    return text_content

def split_into_sections(text_pages):
    """Given a list of page strings, attempt to divide the paper by section headings.

    This is a simple heuristic implementation that looks for lines that appear to be
    section headings, e.g. starting with a digit and a period such as "1. Introduction"
    or "2.1 Method". It returns an ordered list of (heading, content) tuples.
    """
    import re

    # Join pages so that section headings split across page boundaries are handled
    full_text = "\n".join(text_pages)
    # split into lines for inspection
    lines = full_text.splitlines()

    sections = []
    current_heading = None
    current_content_lines = []

    # simple regex for a numbered heading at the start of a line
    # require a terminating period after the numeric part and an uppercase letter
    # for the first word of the title to avoid matching things like "12 hrs".
    # also insist the numeric prefix start with 1-9 (so "0.05..." isn't treated as a
    # section heading) and allow additional sub‑sections like "2.1.3".
    heading_regex = re.compile(r"^\s*([1-9]\d*(?:\.\d+)*)\.\s+([A-Z][A-Za-z0-9\- ]+)")

    # some papers use unnumbered headings; list a few keywords we care about
    keyword_headings = [
        "Introduction",
        "Related Work",
        "Background",
        "Method",
        "Methodology",
        "Model",
        "Approach",
        "Experiments",
        "Results",
        "Discussion",
        "Conclusion",
        "References",
    ]
    keyword_regex = re.compile(r"^\s*(?:" + "|".join(keyword_headings) + r")(?:\s|$)", re.IGNORECASE)

    for line in lines:
        m = heading_regex.match(line)
        key_m = keyword_regex.match(line)
        if m or (key_m and current_heading is None):
            # if we were collecting a previous section, store it
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_content_lines).strip()))
            if m:
                current_heading = f"{m.group(1)} {m.group(2)}"
            else:
                # unnumbered keyword heading
                current_heading = key_m.group(0).strip()
            current_content_lines = []
        else:
            if current_heading is not None:
                current_content_lines.append(line)

    # add the final section
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_content_lines).strip()))

    return sections


