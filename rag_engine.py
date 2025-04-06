from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Global embedding & LLM setup
embedding_model = OpenAIEmbeddings()
llm = OpenAI()

def create_vectorstore(text: str):
    # 1. Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # 2. Embed and create FAISS DB
    vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
    return vectorstore

def get_answer_from_vectorstore(vectorstore, query: str) -> str:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain.run(query)
