import os
import json
import pandas as pd

# Set paths
base_dir = os.path.join("data", "nfcorpus")
output_dir = os.path.join("data", "retrieval")
longform_output_dir = os.path.join("data", "long-form")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(longform_output_dir, exist_ok=True)

# File paths
doc_files = {
    "train": os.path.join(base_dir, "train.docs"),
    "dev": os.path.join(base_dir, "dev.docs"),
    "test": os.path.join(base_dir, "test.docs")
}
query_file = os.path.join(base_dir, "test.all.queries")

# Output paths
json_outputs = {
    "train": os.path.join(output_dir, "nfcorpus_train.json"),
    "dev": os.path.join(output_dir, "nfcorpus_dev.json"),
    "test": os.path.join(output_dir, "nfcorpus_test.json")
}
csv_outputs = {
    "train": os.path.join(output_dir, "nfcorpus_train.csv"),
    "dev": os.path.join(output_dir, "nfcorpus_dev.csv"),
    "test": os.path.join(longform_output_dir, "nfcorpus_test.csv")
}

# Prompt
system_prompt = "You are a multilingual medical assistant. Provide accurate explanations for medical questions in the detected language."

# Process docs
for split in ["train", "dev", "test"]:
    df = pd.read_csv(doc_files[split], sep="\t", names=["id", "text"])
    conversations = []
    csv_data = []

    for _, row in df.iterrows():
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["text"]}  # Dummy answer
            ]
        }
        conversations.append(conversation)

        row_dict = {
            "question": row["text"],
            "answer": row["text"]
        }
        if split in ["train", "dev"]:
            row_dict["id"] = row["id"]
        csv_data.append(row_dict)

    # Write JSON
    with open(json_outputs[split], "w") as jf:
        for entry in conversations:
            json.dump(entry, jf)
            jf.write("\n")

    # Write CSV
    pd.DataFrame(csv_data).to_csv(csv_outputs[split], index=False)

# Process test queries separately
query_json_output = os.path.join(output_dir, "nfcorpus_test.json")
query_csv_output = os.path.join(longform_output_dir, "nfcorpus_test.csv")

query_df = pd.read_csv(query_file, sep="\t", names=["id", "text"])
conversations = []
csv_data = []

for _, row in query_df.iterrows():
    conversation = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": row["text"]}  # Dummy answer
        ]
    }
    conversations.append(conversation)
    csv_data.append({"question": row["text"], "answer": row["text"]})

with open(query_json_output, "w") as jf:
    for entry in conversations:
        json.dump(entry, jf)
        jf.write("\n")

pd.DataFrame(csv_data).to_csv(query_csv_output, index=False)

print(f"Total entries in test queries: {len(conversations)}")
print(f"Saved: {json_outputs['train']}, {csv_outputs['train']}, {json_outputs['dev']}, {csv_outputs['dev']}, {query_json_output}, {query_csv_output}")
