import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class LegalAIAssistant:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", llm_model="google/flan-t5-small", device=-1):
        self.embed_model = SentenceTransformer(embedding_model)
        self.pipe = pipeline("text2text-generation", model=llm_model, device=device, max_length=512)

    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_model.encode(texts)

    @staticmethod
    def get_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_provisions(self, question, provisions, provision_embeddings, top_k=3):
        q_emb = self.get_embeddings(question)[0]
        similarities = [self.get_similarity(q_emb, emb) for emb in provision_embeddings]
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [provisions[i] for i in top_idx]

    def generate_answer(self, question, provisions, provision_embeddings, top_k=3):
        context_provisions = self.retrieve_provisions(question, provisions, provision_embeddings, top_k)
        context = "\n".join(context_provisions)
        prompt = f"""Answer the following question based on the context.
Context:
{context}

Question:
{question}"""
        
        result = self.pipe(prompt)
        answer = result[0].get("generated_text", result[0].get("text", ""))
        return answer, context

    @staticmethod
    def load_dataset(parquet_path):
        df = pd.read_parquet(parquet_path)[["provision", "embedding"]]
        return df
