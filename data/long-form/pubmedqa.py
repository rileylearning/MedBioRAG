import os
import json
import pandas as pd

# Define file paths
train_input_path = os.path.join("data", "pubmedqa", "ori_pqaa.json")
test_input_path = os.path.join("data", "pubmedqa", "ori_pqal.json")
output_dir = os.path.join("data", "long-form")
os.makedirs(output_dir, exist_ok=True)

# Output files
train_jsonl_path = os.path.join(output_dir, "pubmedqa_train.jsonl")
train_csv_path = os.path.join(output_dir, "pubmedqa_train.csv")
test_jsonl_path = os.path.join(output_dir, "pubmedqa_test.jsonl")
test_csv_path = os.path.join(output_dir, "pubmedqa_test.csv")

# Prompt template
system_prompt = "You are a multilingual medical assistant. Answer only with 'yes', 'maybe', or 'no'. Do not add anything else."


def process_long_form(data_dict, include_id=True):
    jsonl_data = []
    csv_data = []

    for idx, (key, item) in enumerate(data_dict.items()):
        if item.get("final_decision") != "yes":
            continue

        context = " ".join(item.get("CONTEXTS", []))
        question = item.get("QUESTION", "")
        long_answer = item.get("LONG_ANSWER", "")

        # JSONL format
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": long_answer}
            ]
        })

        # CSV format
        csv_row = {
            "question": question,
            "answer": long_answer
        }
        if include_id:
            csv_row["id"] = idx + 1
        csv_data.append(csv_row)

    return jsonl_data, csv_data


# ----------- Train Set -----------
with open(train_input_path, "r") as f:
    train_data = json.load(f)

train_jsonl, train_csv = process_long_form(train_data, include_id=True)

# Save JSONL
with open(train_jsonl_path, "w") as f:
    for entry in train_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save CSV
pd.DataFrame(train_csv).to_csv(train_csv_path, index=False)

print(f"Saved {len(train_jsonl)} train entries to:")
print(f"- {train_jsonl_path}")
print(f"- {train_csv_path}")


# ----------- Test Set -----------
with open(test_input_path, "r") as f:
    test_data = json.load(f)

test_jsonl, test_csv = process_long_form(test_data, include_id=False)

# Save JSONL
with open(test_jsonl_path, "w") as f:
    for entry in test_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save CSV (no ID)
pd.DataFrame(test_csv)[["question", "answer"]].to_csv(test_csv_path, index=False)

print(f"Saved {len(test_jsonl)} test entries to:")
print(f"- {test_jsonl_path}")
print(f"- {test_csv_path}")