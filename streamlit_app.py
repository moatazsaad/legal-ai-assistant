import streamlit as st
import numpy as np
from app.legal_assistant import LegalAIAssistant

st.set_page_config(page_title="Legal AI Assistant", layout="wide")

st.title("⚖️ Legal AI Assistant")
st.caption("Ask questions about contract clauses and get clear answers with supporting evidence.")

# Load model + data (cached)
@st.cache_resource
def load_assistant():
    assistant = LegalAIAssistant(llm_model="Qwen/Qwen2.5-0.5B-Instruct")
    df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    embeddings = np.stack(df["embedding"].to_numpy())
    return assistant, df, embeddings

assistant, df, embeddings = load_assistant()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
question = st.text_input("Enter your legal question:")
st.caption("Try: Which law governs this agreement?")

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask")

# Clean LLM output
def clean_answer(answer):
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]
    answer = answer.strip().split("\n")[0]
    return answer

# Ask logic
if ask and question.strip():
    with st.spinner("Generating answer..."):
        answer, retrieved_items = assistant.generate_answer(
            question,
            df,
            embeddings
        )

        # Clean output
        answer = clean_answer(answer)

        # Format context nicely
        context = assistant.format_retrieved_context(retrieved_items)

        st.session_state.history.append((question, answer, context))

elif ask:
    st.warning("Please enter a valid question.")

# Display history
for q, a, c in reversed(st.session_state.history):
    st.markdown(f"### ❓ {q}")
    st.markdown(f"**Answer:** {a}")

    with st.expander("📄 Supporting Provisions"):
        st.code(c)

    st.divider()