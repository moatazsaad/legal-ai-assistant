# Legal AI Assistant

## Overview
The Legal AI Assistant is an end-to-end AI/ML application that retrieves relevant legal provisions and generates concise answers. The project demonstrates expertise in embeddings, LLMs, FastAPI, Streamlit, and cloud deployment.

## Key Features
- Retrieves relevant legal provisions using sentence embeddings and cosine similarity
- Generates concise answers using Flan-T5
- Interactive web interface via **Streamlit**
- REST API using **FastAPI**
- Fully deployed on **Google Cloud Platform (GCP)** with Docker and Kubernetes

## Tech Stack
- **Programming:** Python
- **ML/AI:** Flan-T5, SentenceTransformers embeddings, cosine similarity
- **Frameworks & Libraries:** FastAPI, Streamlit, Pandas, NumPy, Transformers, python-multipart
- **Deployment & Cloud:** Docker, Kubernetes (GKE), GCP

## Project Structure
```

legal-ai-assistant/
├── fastapi\_app.py         # FastAPI backend
├── legal\_assistant.py     # Core AI/ML logic
├── streamlit\_app.py       # Streamlit frontend
├── templates/
│   └── index.html         # HTML template for FastAPI app
├── Dockerfile             # Docker image setup
├── requirements.txt       # Python dependencies
├── deployment.yaml        # Kubernetes deployment config
├── service.yaml           # Kubernetes service config
└── README.md

````

## Getting Started (Local Run)

1. **Clone the repository:**  
```bash
git clone <repo-link>
cd legal-ai-assistant
````

2. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Run FastAPI locally:**

```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

4. **Run Streamlit locally:**

```bash
streamlit run streamlit_app.py
```

## Deployment

* Dockerized app for containerized deployment
* Kubernetes deployment (`deployment.yaml`) and service (`service.yaml`) for GCP
* Live deployment available via Streamlit and FastAPI endpoints

## Demo & Resources

* \[Live Demo (Streamlit)]\([Streamlit Link](https://legal-assistant-9ebcqkryukwaww6c4fgpgv.streamlit.app/))
* \[Video Demo]\([YouTube Link](https://youtu.be/2pU2D58i4Po))

## Project Highlights

* Demonstrates full AI/ML pipeline: dataset processing → embeddings → LLM integration → web interface → cloud deployment
* Combines **FastAPI** and **Streamlit** for both API and interactive frontend
* Scalable cloud-deployed solution using **Docker** and **Kubernetes**

