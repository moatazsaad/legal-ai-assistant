from app.legal_assistant import LegalAIAssistant
from evaluation.eval_data import get_eval_data

# Function to compute Mean Reciprocal Rank
def mean_reciprocal_rank(assistant, df, provision_embeddings, eval_df, k=5):
    """
    Compute MRR using section matching.
    """
    # Store reciprocal rank values
    reciprocal_ranks = []

    # Store detailed results
    results = []

    # Loop through evaluation questions
    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_section = str(row["expected_section"]).strip().lower()

        # Retrieve top-K provisions
        retrieved = assistant.retrieve_provisions(
            question,
            df,
            provision_embeddings,
            top_k=k
        )
#         retrieved = [
#     {
#         "section": "governing laws",
#         "provision": "The parties agree to submit to the jurisdiction of courts in Michigan.",
#         "similarity": 0.91
#     },
#     {
#         "section": "governing laws",
#         "provision": "This agreement shall be governed by the laws of the State of California.",
#         "similarity": 0.88
#     },
#     {
#         "section": "notices",
#         "provision": "All notices must be delivered in writing to the specified address.",
#         "similarity": 0.74
#     }
# ]

        # Extract section names
        retrieved_sections = [str(item["section"]).strip().lower() for item in retrieved]

        # Default: not found
        found_rank = None

        # Look for the first correct rank
        for rank, section in enumerate(retrieved_sections, start=1):
            if section == expected_section:
                found_rank = rank
                break

        # Reciprocal rank = 1 / rank if found, else 0
        reciprocal_rank = 1 / found_rank if found_rank is not None else 0.0

        reciprocal_ranks.append(reciprocal_rank)

        # Save details
        results.append({
            "question": question,
            "expected_section": expected_section,
            "retrieved_sections": retrieved_sections,
            "first_correct_rank": found_rank,
            "reciprocal_rank": reciprocal_rank
        })

    # Compute average
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mrr, results

if __name__ == "__main__":
    assistant = LegalAIAssistant()
    df = assistant.load_dataset("hf://datasets/Moataz88Saad/ledgar_qa_retrieval/dataset.parquet")
    provision_embeddings = assistant.prepare_embeddings(df)
    eval_df = get_eval_data()
    mrr, results = mean_reciprocal_rank(assistant, df, provision_embeddings, eval_df, k=5)

    print(f"\nMRR@5: {mrr:.4f}")
    print("\nDetailed MRR results:\n")
    for item in results:
        print(item)