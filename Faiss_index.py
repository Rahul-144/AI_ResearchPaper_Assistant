from parser import load_pdf
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
import os

text = load_pdf("/Users/rahulbiju/Downloads/241213769v1_250306_120359.pdf")
text = "\n".join(text)

# Step 2: Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(text)
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
    print("Creating new in-memory index")
    
    # 3. Create the FAISS index using the cached embedder
    # from_texts is the correct method for multiple strings
    vectorstore = FAISS.from_texts(
        texts=texts, 
        embedding=cached_embeddings, 
        
    )
    
    vectorstore.index.nprobe = 10
    
    # Do NOT call .save_local() here
    return vectorstore