from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from Faiss_index import faiss_index
import os
from  dotenv import load_dotenv
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=4)

load_dotenv()
vectorstore = faiss_index()
def RAG_Engine(query):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    relevant_chunks= compression_retriever.invoke(query)

    # Step 2: Retrieve relevant chunks based on the query
    # relevant_chunks = vectorstore.similarity_search(query, k=5)
    
    # Step 3: Generate an answer using the retrieved chunks
    llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("OPEN_API_KEY"),
    model_name="qwen/qwen3-32b",
    temperature=0.2 # Valid Groq model'
   
)
    prompt_template = """
    You are a helpful research assistant.

Use the retrieved information to answer the question.

Rules:

Answer strictly based on the retrieved information.

You may combine information from multiple chunks.

You may paraphrase or infer the objective if it is clearly implied by the retrieved text.

If the answer cannot be determined from the retrieved information, say: "I don't know."

Return only the final answer. Do not show reasoning.

Retrieved Information:
{retrieved_info}

Question:
{question}

Answer:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    # 2. Prepare your context (retrieved_info)
    retrieved_info = "\n\n".join(
    f"[{c.metadata.get('section','')}] {c.page_content}"
    for c in relevant_chunks
)
    # 3. Execute the chain once with all variables
    # Use .invoke() instead of .run() (which is deprecated)
    answer = chain.invoke({
        "retrieved_info": retrieved_info, 
        "question": query
    })

    return answer
