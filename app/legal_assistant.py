import re
import warnings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

class LegalAIAssistant:
    # def __init__(self, embedding_model="all-MiniLM-L6-v2", llm_model="google/flan-t5-base", device=-1):
    def __init__(self, embedding_model="all-MiniLM-L6-v2", llm_model="google/flan-t5-small", device=-1):

        # Load embedding model
        try:
            self.embed_model = SentenceTransformer(embedding_model)
        except Exception as e:
            raise RuntimeError(f"Error loading embedding model: {e}")

        # Load LLM model
        try:
            self.pipe = pipeline(
                "text2text-generation",
                # "text-generation",
                model=llm_model,
                device=device,
                max_new_tokens=512
            )
        except Exception as e:
            raise RuntimeError(f"Error loading LLM model: {e}")

        # Keywords to help connect user questions to section names in dataset
        self.section_keywords = {
            "governing laws": ["governing law", "jurisdiction", "court", "venue", "state law"],
            "notices": ["notice", "notices", "notify", "written notice"],
            "entire agreements": ["entire agreement", "oral modification", "amendment", "waiver"],
            "counterparts": ["counterpart", "counterparts"],
            "severability": ["severability", "unenforceable", "invalid provision"],
            "general": ["general", "miscellaneous"],
            "taxes": ["tax", "taxes", "withholding", "tax consequences"],
            "insurances": ["insurance", "insured", "premiums", "coverage"],
            "confidentiality": ["confidential", "confidentiality", "nda", "non-disclosure"],
            "capitalization": ["capital stock", "shares", "stock", "equity"]
        }

    def get_embeddings(self, texts):
        # Turn one text into a list so the model can process it
        if isinstance(texts, str):
            texts = [texts]

        try:
            return self.embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            raise RuntimeError(f"Error computing embeddings: {e}")

    def clean_text(self, text):
        # Lowercase and remove extra spaces
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def infer_sections_from_question(self, question):
        # Try to guess which legal section the user is asking about
        q = self.clean_text(question)
        matched_sections = []

        for section, keywords in self.section_keywords.items():
            for keyword in keywords:
                if keyword in q:
                    matched_sections.append(section)
                    break

        return matched_sections

    def compute_similarities(self, query_embedding, provision_embeddings):
        # Since embeddings are normalized, cosine similarity is just dot product
        if provision_embeddings.size == 0:
            return np.array([])
        return provision_embeddings @ query_embedding

    def score_provisions(self, question, df, provision_embeddings):
        # Get question embedding
        q_emb = self.get_embeddings(question)[0]

        # Compute similarity between question and all provisions
        similarities = self.compute_similarities(q_emb, provision_embeddings)

        ranked_df = df.copy()
        ranked_df["similarity"] = similarities
        ranked_df["section_boost"] = 0.0
        ranked_df["final_score"] = ranked_df["similarity"]

        # Boost rows if section seems related to question
        inferred_sections = self.infer_sections_from_question(question)
        if inferred_sections:
            ranked_df["section_boost"] = ranked_df["section"].apply(
                lambda x: 0.10 if self.clean_text(x) in inferred_sections else 0.0
            )
            ranked_df["final_score"] = ranked_df["similarity"] + ranked_df["section_boost"]

        # Sort by final score
        ranked_df = ranked_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        return ranked_df

    def retrieve_provisions(self, question, df, provision_embeddings, top_k=3):
        # Retrieve top relevant provisions
        ranked_df = self.score_provisions(question, df, provision_embeddings)

        results = []
        for _, row in ranked_df.head(top_k).iterrows():
            results.append({
                "provision": row["provision"],
                "section": row["section"],
                "similarity": float(row["similarity"]),
                "final_score": float(row["final_score"])
            })

        return results

    def build_prompt(self, question, retrieved_items):
        # Build a prompt using the retrieved provisions
        context_parts = []
        for i, item in enumerate(retrieved_items, start=1):
            context_parts.append(
                f"[Provision {i}]\nSection: {item['section']}\nText: {item['provision']}"
            )

        context = "\n\n".join(context_parts)
        prompt = f"""
        
        You are a legal assistant.

        Answer clearly in plain English.
        Do not copy the text directly.
        Paraphrase the answer in a natural sentence to sound natural.
        If the answer is not clearly supported, say: "The context is insufficient."

        Provisions:
        {context}

        Question:
        {question}

        Answer:
        """.strip()

        return prompt

    def generate_answer(self, question, df, provision_embeddings, top_k=3):
        # Generate answer from retrieved provisions
        if not question.strip():
            return "Please enter a valid question.", []

        retrieved_items = self.retrieve_provisions(question, df, provision_embeddings, top_k=top_k)

        if not retrieved_items:
            return "No relevant provisions found.", []

        prompt = self.build_prompt(question, retrieved_items)

        try:
            result = self.pipe(prompt)
            answer = result[0].get("generated_text", result[0].get("text", "")).strip()
            # answer = result[0]["generated_text"].strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return answer, retrieved_items

    def load_dataset(self, parquet_path):
        # Load dataset
        try:
            df = pd.read_parquet(parquet_path)[["provision", "section", "embedding"]]
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {parquet_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

        if df.empty:
            raise ValueError("Dataset is empty.")

        # Remove missing rows
        df = df.dropna(subset=["provision", "section", "embedding"]).reset_index(drop=True)
        return df

    def prepare_embeddings(self, df):
        # Convert embeddings column into a matrix
        try:
            embeddings = np.stack(df["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).to_numpy())
        except Exception as e:
            raise RuntimeError(f"Error preparing embeddings: {e}")

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embeddings = embeddings / norms
        return embeddings

    def format_retrieved_context(self, retrieved_items):
        # Show supporting provisions clearly
        lines = []
        for i, item in enumerate(retrieved_items, start=1):
            lines.append(
                f"[Provision {i}] Section: {item['section']} | "
                f"Similarity: {item['similarity']:.4f}\n"
                f"{item['provision']}\n"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    assistant = LegalAIAssistant()

    # Load dataset
    try:
        df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    except Exception as e:
        print(e)
        exit()

    # Prepare embeddings once
    try:
        provision_embeddings = assistant.prepare_embeddings(df)
    except Exception as e:
        print(e)
        exit()

    # Ask questions in terminal
    while True:
        question = input("\nEnter your legal question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        answer, retrieved_items = assistant.generate_answer(
            question,
            df,
            provision_embeddings,
            top_k=3
        )

        print("\n=== Answer ===")
        print(answer)

        print("\n=== Supporting Provisions ===")
        print(assistant.format_retrieved_context(retrieved_items))