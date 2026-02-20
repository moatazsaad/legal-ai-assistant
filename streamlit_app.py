import streamlit as st
import numpy as np
from legal_assistant import LegalAIAssistant

st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("⚖️ Legal AI Assistant")
st.write("Ask questions about legal provisions and get concise AI-generated answers with context.")

@st.cache_resource
def load_assistant():
    assistant = LegalAIAssistant()
    df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    # Convert saved embeddings to numpy arrays if needed
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    return assistant, df

assistant, df = load_assistant()

# Store chat history
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Enter your legal question:")

if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer, context = assistant.generate_answer(
            question, df["provision"].tolist(), np.stack(df["embedding"].to_numpy())
        )
        st.session_state.history.append((question, answer, context))

# Display history
for q, a, c in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown(f"**Context:** {c}")
    st.markdown("---")