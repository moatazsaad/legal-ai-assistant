# Legal AI Assistant

A deployed Retrieval-Augmented Generation (RAG) system for answering questions over legal contract clauses, combining semantic retrieval, LLM generation, evaluation, and cloud deployment.

---

## Overview
This project builds an end-to-end legal question answering system that retrieves relevant contract provisions and generates clear, evidence-based answers.

It demonstrates practical AI engineering across:
- Retrieval-augmented generation (RAG)
- Semantic search using embeddings
- Evaluation with ranking metrics
- Backend API development
- Cloud deployment (AWS and GCP)

---

## Key Features
- Semantic retrieval using sentence embeddings and similarity scoring  
- LLM-based answer generation (Flan-T5)  
- Transparent outputs with supporting provisions  
- Evaluation pipeline with strong metrics  
- FastAPI backend and Streamlit UI  
- Deployment using Docker, Kubernetes, AWS EC2, and GCP  

---

## How It Works
1. Convert user query into embeddings  
2. Retrieve top-k similar legal clauses  
3. Pass retrieved context to the LLM  
4. Generate answer grounded in retrieved provisions  
5. Return answer with supporting evidence  

---

## Architecture

```

User Query
↓
Embedding Model
↓
Similarity Search (Top-K)
↓
Retrieved Provisions
↓
LLM Generation
↓
Final Answer + Evidence

```

---

## Evaluation

The retrieval system was evaluated on a curated dataset of 20 legal questions.

| Metric     | Score |
|------------|------|
| Recall@3   | 0.95 |
| MRR@5      | 0.925 |

### Interpretation
- The correct section appears within the top 3 results for 95% of queries  
- In most cases, the correct section is ranked first  

### Example
Question: Which law governs this agreement?  
Top Results: governing laws, governing laws, severability  
Result: Correct section ranked first  

### Failure Case
Question: Does the company indemnify parties for tax penalties?  
Result: Incorrect sections retrieved, showing realistic system limitations  

---

## Tech Stack

- Programming: Python  
- ML/AI: SentenceTransformers, Flan-T5, Qwen  
- Frameworks: FastAPI, Streamlit  
- Libraries: Pandas, NumPy, Transformers  
- Deployment: Docker, Kubernetes (GKE), AWS EC2  

---

## Project Structure

```

legal-ai-assistant/
├── app/
│   ├── fastapi_app.py
│   └── legal_assistant.py
├── evaluation/
│   ├── eval_mrr.py
│   ├── eval_recall.py
├── templates/
│   └── index.html
├── streamlit_app.py
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── requirements.txt
└── README.md

````

---

## Run Locally

```bash
git clone https://github.com/moatazsaad/legal-ai-assistant.git
cd legal-ai-assistant

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

uvicorn app.fastapi_app:app --reload --port 8000
streamlit run streamlit_app.py
````

Access:

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* UI: [http://localhost:8501](http://localhost:8501)

---

## Run with Docker

```bash
docker build -t legal-ai-assistant .
docker run -p 8000:8000 legal-ai-assistant
```

---

## Deployment

* Deployed on AWS EC2 and Google Cloud (GKE)
* Containerized using Docker
* Supports scalable deployment via Kubernetes

---

## Example Questions

* Which law governs this agreement?
* How must notices be given?
* What happens if a provision is unenforceable?
* Who is responsible for insurance premiums?

---

## Key Highlights

* End-to-end RAG system (retrieval, generation, evaluation, deployment)
* Explainable AI with evidence-backed outputs
* Real evaluation using IR metrics (Recall, MRR)
* Designed for production-style deployment

---

