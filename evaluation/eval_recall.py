from app.legal_assistant import LegalAIAssistant
from evaluation.eval_data import get_eval_data

def recall_at_k(assistant, df, provision_embeddings, eval_df, k=3):
    """
    Compute Recall@K using section matching.
    A query is counted as correct if the expected section appears
    anywhere in the top-K retrieved results.
    """
    hits = 0
    results = []

    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_section = str(row["expected_section"]).strip().lower()

        retrieved = assistant.retrieve_provisions(
            question,
            df,
            provision_embeddings,
            top_k=k
        )

        retrieved_sections = [
            str(item["section"]).strip().lower() for item in retrieved
        ]

        hit = expected_section in retrieved_sections
        if hit:
            hits += 1

        results.append({
            "question": question,
            "expected_section": expected_section,
            "retrieved_sections": retrieved_sections,
            "hit@k": hit
        })

    recall = hits / len(eval_df) if len(eval_df) > 0 else 0.0
    return recall, results


if __name__ == "__main__":
    assistant = LegalAIAssistant()

    df = assistant.load_dataset(
        "hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet"
    )

    provision_embeddings = assistant.prepare_embeddings(df)
    eval_df = get_eval_data()

    recall, results = recall_at_k(
        assistant,
        df,
        provision_embeddings,
        eval_df,
        k=3
    )

    print(f"\nRecall@3: {recall:.3f}")
    print("\nDetailed Recall results:\n")
    for item in results:
        print(item)