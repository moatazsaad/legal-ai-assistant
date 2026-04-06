from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.legal_assistant import LegalAIAssistant
import asyncio

app = FastAPI(title="Legal AI Assistant API")

# Load assistant and dataset once when app starts
print("Loading assistant...")
assistant = LegalAIAssistant()

print("Loading dataset...")
df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")

print("Preparing embeddings...")
provision_embeddings = assistant.prepare_embeddings(df)

print("Startup complete.")

# Templates
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "context": "",
            "question": ""
        }
    )


# Ask question
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    if not question.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "Please enter a valid question.",
                "context": "",
                "question": question
            }
        )

    try:
        # Run model call in background thread so FastAPI does not block
        loop = asyncio.get_event_loop()
        answer, retrieved_items = await loop.run_in_executor(
            None,
            assistant.generate_answer,
            question,
            df,
            provision_embeddings,
            3
        )

        # Format supporting provisions for display on page
        context = assistant.format_retrieved_context(retrieved_items)

    except Exception as e:
        answer = f"An error occurred: {e}"
        context = ""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "context": context,
            "question": question
        }
    )