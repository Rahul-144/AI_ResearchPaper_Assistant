from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from Faiss_index import faiss_index
import os
from  dotenv import load_dotenv
load_dotenv()
def RAG_Engine(query):
    # Step 1: Create the FAISS index
    vectorstore = faiss_index()
    
    # Step 2: Retrieve relevant chunks based on the query
    relevant_chunks = vectorstore.similarity_search(query, k=5)
    
    # Step 3: Generate an answer using the retrieved chunks
    llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("OPEN_API_KEY"),
    model_name="qwen/qwen3-32b" # Valid Groq model'
   
)
    
    prompt_template = """
    You are a helpful assistant. Use the following retrieved information to answer the question:
    
    Retrieved Information:
    {retrieved_info}
    
    Question:
    {question}
    
    Answer:
    Rules:
- Use only the retrieved information to answer the question.
- If the retrieved information does not contain the answer, say "I don't know."

    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    retrieved_info = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    answer = chain.run(retrieved_info=retrieved_info, question=query)
    
    return answer
