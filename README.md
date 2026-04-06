# Legal AI Assistant

A production style **Retrieval-Augmented Generation (RAG)** system for answering questions over legal contract clauses, built with **FastAPI**, **Docker**, and **AWS EC2**.

## Live Demo (may be temporarily offline to reduce costs)

- **App:** `http://54.167.63.246:8000`
- **API Docs:** `http://54.167.63.246:8000/docs`

## Overview

This project implements an end-to-end **legal question answering system** that retrieves relevant contract provisions and generates answers grounded in those provisions.

It demonstrates practical AI engineering skills including:

- RAG pipeline design
- semantic search with embeddings
- retrieval evaluation using ranking metrics
- API deployment with FastAPI, Docker, and AWS

## Features

- Semantic retrieval over legal clauses
- Retrieval-augmented answer generation
- FastAPI backend with a simple web interface
- Evaluation pipeline with **Recall@K** and **MRR@K**
- Cloud deployment on AWS EC2

## How It Works

1. **Load dataset**  
   Legal clauses are loaded from a structured dataset.

2. **Create embeddings**  
   Each clause is converted into vector embeddings for semantic search.

3. **Retrieve relevant clauses**  
   The system finds the top-k most relevant provisions for a user query.

4. **Generate answer**  
   The retrieved provisions are used as supporting context for the final answer.

## Architecture

```text
User Query
   ↓
FastAPI App
   ↓
Embedding Model
   ↓
Similarity Search (Top-K Retrieval)
   ↓
Retrieved Legal Clauses
   ↓
Generated Answer
````

## Evaluation Results

The retrieval component was evaluated on a curated 20-question legal QA dataset using section-level ground truth.

* **Recall@3: 0.950**
* **MRR@5: 0.9250**

### Interpretation

* The correct section appears within the top 3 results in **95% of queries**
* In most cases, the correct section is ranked **first**, with occasional cases at rank 2

### Example Retrieval

* **Question:** Which law governs this agreement?
* **Expected Section:** governing laws
* **Top Retrieved Sections:** governing laws, governing laws, severability
* **Result:** Correct section ranked first

### Challenging Case

* **Question:** What legal authority resolves disputes in this agreement?
* **Expected Section:** governing laws
* **Top Retrieved Sections:** general, governing laws, governing laws
* **Result:** Correct section found at rank 2

### Failure Case

* **Question:** Does the company indemnify parties for tax penalties?
* **Expected Section:** general
* **Top Retrieved Sections:** taxes, taxes, taxes
* **Result:** Correct section not retrieved in top 3

## Project Structure

```text
legal-assistant/
│
├── app/
│   ├── fastapi_app.py
│   └── legal_assistant.py
├── evaluation/
│   ├── eval_mrr.py
│   └── eval_recall.py
├── templates/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Tech Stack

* Python
* FastAPI
* Sentence Transformers
* PyTorch
* Pandas
* NumPy
* Docker
* AWS EC2

## Run Locally

```bash
git clone <your-repo-url>
cd legal-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.fastapi_app:app --reload --port 8000
```

For Windows:

```bash
venv\Scripts\activate
```

Then open:

```text
http://127.0.0.1:8000
```

## Run with Docker

```bash
docker build -t legal-ai-assistant .
docker run -p 8000:8000 legal-ai-assistant
```

## Example Questions

* Who is responsible for paying insurance premiums and what proof may be required?
* Which law governs this agreement?
* How must notices be given under this agreement?
* What happens if a provision is unenforceable?
* Does this agreement override prior oral agreements?



