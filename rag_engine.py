from parser import load_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# import os
# import getpass

# # Set OpenAI key (only needed if using OpenAI LLM)
# if not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Step 1: Load PDF and Extract Text
def rag():
    text = load_pdf("/Users/rahulbiju/Downloads/241213769v1_250306_120359.pdf")
    text = "\n".join(text)

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Step 3: Embeddings (Local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 4: Create Vector DB
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    model_id = "google/flan-t5-base"  # You can also use "google/flan-t5-base" for CPU

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")  # Will use GPU if available

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

    llm = HuggingFacePipeline(pipeline=pipe) 

    # Step 6: QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"verbose": False}  
    )
    return qa
