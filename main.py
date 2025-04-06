from fastapi import FastAPI, UploadFile, File
from parser import extract_text_from_pdf
from rag_engine import create_vectorstore, get_answer_from_vectorstore

import os

app = FastAPI()
vectorstore_cache = {}  # temp memory for testing

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_papers/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        text = extract_text_from_pdf(file_path)
        vs = create_vectorstore(text)
        vectorstore_cache[file.filename] = vs
        return {"message": "PDF uploaded and indexed successfully.", "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/")
async def ask_question(filename: str, question: str):
    if filename not in vectorstore_cache:
        return {"error": "PDF not uploaded or indexed yet."}
    vs = vectorstore_cache[filename]
    answer = get_answer_from_vectorstore(vs, question)
    return {"answer": answer}
