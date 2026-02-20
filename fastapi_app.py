from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from legal_assistant import LegalAIAssistant
import asyncio


app = FastAPI(title="Legal AI Assistant API")

# Load assistant and dataset once
assistant = LegalAIAssistant()
df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
provisions = df["provision"].tolist()
provision_embeddings = np.stack(df["embedding"].to_numpy())

# Templates
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

# Ask question
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    if not question.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": "Please enter a valid question.",
            "context": "",
            "question": question
        })

    try:
        loop = asyncio.get_event_loop()
        answer, context = await loop.run_in_executor(
            None, assistant.generate_answer, question, provisions, provision_embeddings, 3
        )
    except Exception as e:
        answer = f"An error occurred: {e}"
        context = ""
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": answer,
        "context": context,
        "question": question
    })
