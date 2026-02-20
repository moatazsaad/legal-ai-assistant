from datasets import load_dataset

# Correct repo path
ds = load_dataset("Moataz88Saad/ledgar_qa_retrieval")

# Convert first split to pandas
df = ds[list(ds.keys())[0]].to_pandas()

print("COLUMNS:", df.columns.tolist())
print("\nFIRST 3 ROWS:")
print(df.head(10))

# Print first 5 provisions
print("Sample provisions:\n")
for i, prov in enumerate(df['provision'].head(10), start=1):
    print(f"{i}. {prov}\n")

# Where is the location of the courts that have jurisdiction over disputes under this Agreement?

