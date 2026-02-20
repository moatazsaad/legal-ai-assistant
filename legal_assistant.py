import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class LegalAIAssistant:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", llm_model="google/flan-t5-small", device=-1):
        # Initialize embedding model and LLM pipeline
        try:
            self.embed_model = SentenceTransformer(embedding_model)
        except Exception as e:
            raise RuntimeError(f"Error loading embedding model: {e}")

        try:
            self.pipe = pipeline("text2text-generation", model=llm_model, device=device, max_new_tokens=512)
        except Exception as e:
            raise RuntimeError(f"Error loading LLM model: {e}")

    def get_embeddings(self, texts):
        # Compute embeddings 
        if isinstance(texts, str):
            texts = [texts]
        try:
            return self.embed_model.encode(texts)
        except Exception as e:
            raise RuntimeError(f"Error computing embeddings: {e}")

    @staticmethod
    def get_similarity(a, b):
        # Compute cosine similarity between two embeddings
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception:
            return 0.0  # fallback if error or shape mismatch

    def retrieve_provisions(self, question, provisions, provision_embeddings, top_k=3):
        # Retrieve top-k provisions most relevant to a question
        q_emb = self.get_embeddings(question)[0]
        if provision_embeddings.size == 0:
            return []
        similarities = [self.get_similarity(q_emb, emb) for emb in provision_embeddings]
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [provisions[i] for i in top_idx if i < len(provisions)]

    def generate_answer(self, question, provisions, provision_embeddings, top_k=3):
        # Generate answer to a question using top-k relevant provisions and LLM
        if not question.strip():
            return "Please enter a valid question.", ""
        context_provisions = self.retrieve_provisions(question, provisions, provision_embeddings, top_k)
        if not context_provisions:
            return "No relevant provisions found.", ""
        context = "\n".join(context_provisions)
        prompt = f"""Answer the following question based on the context.
Context:
{context}

Question:
{question}"""
        try:
            result = self.pipe(prompt)
            answer = result[0].get("generated_text", result[0].get("text", ""))
        except Exception as e:
            answer = f"Error generating answer: {e}"
        return answer, context

    @staticmethod
    def load_dataset(parquet_path):
        # Load provisions and embeddings from a parquet dataset
        try:
            df = pd.read_parquet(parquet_path)[["provision", "embedding"]]
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {parquet_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
        if df.empty:
            raise ValueError("Dataset is empty.")
        return df


if __name__ == "__main__":
    assistant = LegalAIAssistant()

    # Load dataset
    try:
        df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    except Exception as e:
        print(e)
        exit()

    provisions = df["provision"].tolist()
    provision_embeddings = np.stack(df["embedding"].to_numpy())

    # Interactive loop to ask legal questions
    while True:
        question = input("\nEnter your legal question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        answer, context = assistant.generate_answer(question, provisions, provision_embeddings)
        print(f"\nAnswer: {answer}")
        print(f"Context used:\n{context}")
