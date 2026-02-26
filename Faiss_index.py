from parser import load_pdf, split_into_sections
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import InMemoryByteStore
import os

# load the PDF pages
pages = load_pdf("/Users/rahulbiju/Downloads/TrafficFlowGAN_.pdf")

# split into sections and then chunk each section separately
sections = split_into_sections(pages)

# Step 2: Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = []  # will hold the chunk strings
metadatas = []  # optional: store section headings as metadata
for heading, content in sections:
    # chunk the content of each section
    chunks = text_splitter.split_text(content)
    for c in chunks:
        # optionally prefix the heading or keep as metadata
        texts.append(c)
        metadatas.append({"section": heading})

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = InMemoryByteStore()

# 2. Wrap your embeddings
# The namespace helps distinguish between different models/versions in memory
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, 
    store, 
    namespace="in_memory_cache"
)

def faiss_index():
    print("Creating new in-memory index (section-aware chunks)")
    
    # 3. Create the FAISS index using the cached embedder
    # from_texts is the correct method for multiple strings. Pass
    # metadatas so each chunk keeps its original section heading.
    vectorstore = FAISS.from_texts(
        texts=texts, 
        embedding=cached_embeddings, 
        metadatas=metadatas,
    )
    
    # vectorstore.index.nprobe = 10
    
    # Do NOT call .save_local() here
    return vectorstore
