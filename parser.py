
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

