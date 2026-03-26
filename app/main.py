# main.py — FastAPI server with all endpoints

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, os
from app.rag_pipeline import process_pdf, get_retriever
from app.chains import build_qa_chain, ask_question
from app.memory import build_chat_chain, clear_memory

app = FastAPI(title="RAG Chatbot API")

# Global state (for single-user dev setup)
# In production: use a proper session store (Redis, DB)
vectorstore = None


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


# Health check — always build this first, easiest to test
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


# Upload and process a PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    # Save uploaded file temporarily
    data_path = os.getenv("data_path", "data")
    temp_path = os.path.join(data_path, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    vectorstore, err = process_pdf(temp_path, force_rebuild=True)
    if err:
        raise HTTPException(status_code=400, detail=err)

    return {"status": "ok", "filename": file.filename}


# Ask a question
@app.post("/ask")
async def ask(req: QuestionRequest):
    if not vectorstore:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")

    retriever = get_retriever(vectorstore)
    chain = build_chat_chain(retriever, req.session_id)
    result = chain.invoke({"question": req.question})

    return {
        "answer": result["answer"],
        "sources": [
            {"page": d.metadata.get("page"), "snippet": d.page_content[:150]}
            for d in result.get("source_documents", [])
        ],
    }


# Clear session memory
@app.delete("/clear/{session_id}")
def clear(session_id: str):
    clear_memory(session_id)
    return {"status": "cleared"}
