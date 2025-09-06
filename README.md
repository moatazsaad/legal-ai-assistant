# Legal AI Assistant
Ask questions about legal provisions and get concise answers using AI models.

## Project Overview
This project provides an AI-powered assistant for legal documents. It leverages sentence embeddings and large language models (LLMs) to help users quickly find relevant legal provisions and generate concise answers. The system works by embedding the legal text, comparing it with user queries, and using the most relevant context to produce answers.

Key components:
- **Sentence embeddings**: Convert legal provisions and user questions into numerical vectors for similarity search.
- **Similarity search**: Retrieves the top relevant provisions based on query-provision similarity.
- **LLM-based answer generation**: Uses a text-to-text generation model to produce answers using the retrieved provisions as context.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Features
- Retrieve relevant legal provisions
- Generate AI-based answers with context

## Usage Example
```python
from legal_assistant import LegalAIAssistant
import pandas as pd

# Load dataset
df = LegalAIAssistant.load_dataset("provisions.parquet")
provisions = df["provision"].tolist()
provision_embeddings = df["embedding"].tolist()

# Initialize assistant
assistant = LegalAIAssistant()

# Ask a question
question = "What are the termination clauses?"
answer, context = assistant.generate_answer(question, provisions, provision_embeddings)
print("Answer:", answer)
print("Context used:", context)
