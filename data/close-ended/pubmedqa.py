import os
import json
import pandas as pd

# Define input and output paths
train_input_path = os.path.join("data", "pubmedqa", "ori_pqaa.json")
test_input_path = os.path.join("data", "pubmedqa", "ori_pqal.json")
output_dir = os.path.join("data", "close-ended", "pubmedqa")
os.makedirs(output_dir, exist_ok=True)

# Define output file paths
train_jsonl_path = os.path.join(output_dir, "pubmedqa_train.jsonl")
train_csv_path = os.path.join(output_dir, "pubmedqa_train.csv")
test_jsonl_path = os.path.join(output_dir, "pubmedqa_test.jsonl")
test_csv_path = os.path.join(output_dir, "pubmedqa_test.csv")

# Prompt template
system_prompt = "You are a multilingual medical assistant. Answer only with 'yes', 'maybe', or 'no'. Do not add anything else."


def process_pubmedqa(data_dict, include_id=True):
    jsonl_data = []
    csv_data = []

    for idx, (key, entry) in enumerate(data_dict.items()):
        reasoning_required_pred = entry.get("reasoning_required_pred")
        final_decision = entry.get("final_decision")

        # Filter: only entries that require reasoning and have valid decisions
        if reasoning_required_pred != "yes" or final_decision not in ["yes", "no", "maybe"]:
            continue

        # Prepare input for the assistant
        context = " ".join(entry.get("CONTEXTS", []))
        question = entry.get("QUESTION", "")
        user_content = f"{context} {question}".strip()

        # Build JSONL format
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": final_decision}
            ]
        })

        # Build CSV format
        csv_entry = {
            "question": question,
            "answer": final_decision
        }
        if include_id:
            csv_entry["id"] = idx + 1
        csv_data.append(csv_entry)

    return jsonl_data, csv_data


# ---------- Process Train ----------
with open(train_input_path, "r") as f:
    train_data = json.load(f)

train_jsonl, train_csv = process_pubmedqa(train_data, include_id=True)

# Save train.jsonl
with open(train_jsonl_path, "w") as f:
    for entry in train_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save train.csv
pd.DataFrame(train_csv).to_csv(train_csv_path, index=False)

print(f"Saved {len(train_jsonl)} training entries to:")
print(f"- {train_jsonl_path}")
print(f"- {train_csv_path}")


# ---------- Process Test ----------
with open(test_input_path, "r") as f:
    test_data = json.load(f)

test_jsonl, test_csv = process_pubmedqa(test_data, include_id=False)

# Save test.jsonl
with open(test_jsonl_path, "w") as f:
    for entry in test_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save test.csv (without ID)
pd.DataFrame(test_csv)[["question", "answer"]].to_csv(test_csv_path, index=False)

print(f"Saved {len(test_jsonl)} test entries to:")
print(f"- {test_jsonl_path}")
print(f"- {test_csv_path}")