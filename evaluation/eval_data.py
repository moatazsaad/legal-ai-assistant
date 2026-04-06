import pandas as pd

# Function that returns golden dataset for RAG evaluation
def get_eval_data():

    eval_data = [
        {
            "question": "Who is responsible for paying insurance premiums and what proof may be required?",
            "expected_section": "insurances",
            "expected_answer": "The responsible party must pay premiums and provide proof such as receipts upon request."
        },
        {
            "question": "Does the company provide any guarantees about tax consequences?",
            "expected_section": "taxes",
            "expected_answer": "No, the company does not make any warranties about tax consequences."
        },
        {
            "question": "Which courts have jurisdiction over disputes under this agreement?",
            "expected_section": "governing laws",
            "expected_answer": "Federal and state courts specified in the agreement have jurisdiction."
        },
        {
            "question": "Has the company issued any new stock recently?",
            "expected_section": "capitalization",
            "expected_answer": "No new stock has been issued except under employee plans or existing instruments."
        },
        {
            "question": "Can this agreement be executed in multiple counterparts?",
            "expected_section": "counterparts",
            "expected_answer": "Yes, it may be executed in multiple counterparts forming one agreement."
        },
        {
            "question": "Does the agreement provide guarantees regarding Section 409A compliance?",
            "expected_section": "general",
            "expected_answer": "No, it does not guarantee exemption or protection from penalties."
        },
        {
            "question": "Do both parties agree to submit to court jurisdiction?",
            "expected_section": "governing laws",
            "expected_answer": "Yes, both parties agree to submit to jurisdiction."
        },
        {
            "question": "How must notices be given under this agreement?",
            "expected_section": "notices",
            "expected_answer": "Notices must be delivered according to the agreement terms."
        },
        {
            "question": "Which law governs this agreement?",
            "expected_section": "governing laws",
            "expected_answer": "The agreement is governed by the specified state law."
        },
        {
            "question": "Are oral modifications to the agreement valid?",
            "expected_section": "entire agreements",
            "expected_answer": "No, oral modifications are not valid."
        },

        {
            "question": "What happens if a provision of the agreement is unenforceable?",
            "expected_section": "severability",
            "expected_answer": "It may be modified or removed while the rest of the agreement remains valid."
        },
        {
            "question": "Does this agreement override prior agreements?",
            "expected_section": "entire agreements",
            "expected_answer": "Yes, it represents the full agreement and overrides prior ones."
        },
        {
            "question": "What obligations exist regarding maintaining insurance coverage?",
            "expected_section": "insurances",
            "expected_answer": "Parties must maintain valid insurance and ensure premiums are paid."
        },
        {
            "question": "Can notices be sent in writing only?",
            "expected_section": "notices",
            "expected_answer": "Yes, notices must follow written communication requirements."
        },
        {
            "question": "Is the agreement interpreted under a specific state law?",
            "expected_section": "governing laws",
            "expected_answer": "Yes, it is interpreted under a specified jurisdiction’s laws."
        },
        {
            "question": "Can the agreement be signed in separate copies?",
            "expected_section": "counterparts",
            "expected_answer": "Yes, separate copies together form one agreement."
        },
        {
            "question": "Are there restrictions on issuing new shares?",
            "expected_section": "capitalization",
            "expected_answer": "Yes, issuance is limited to specific allowed cases."
        },
        {
            "question": "Does the company indemnify parties for tax penalties?",
            "expected_section": "general",
            "expected_answer": "No, it does not cover penalties or taxes."
        },
        {
            "question": "What legal authority resolves disputes in this agreement?",
            "expected_section": "governing laws",
            "expected_answer": "Specified courts have authority over disputes."
        },
        {
            "question": "Is written consent required for changes to the agreement?",
            "expected_section": "entire agreements",
            "expected_answer": "Yes, changes must be made formally in writing."
        }
    ]

    return pd.DataFrame(eval_data)

if __name__ == "__main__":
    df = get_eval_data()
    print(df)

