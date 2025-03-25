import os
import json
import pandas as pd

# Define input paths
train_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "BioASQ-training12b", "training12b_new.json")
test_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "Task12BGoldenEnriched", "12B4_golden.json")

# Define output directory and paths
output_dir = os.path.join("data", "long-form")
os.makedirs(output_dir, exist_ok=True)

train_jsonl_path = os.path.join(output_dir, "bioasq_train.jsonl")
train_csv_path = os.path.join(output_dir, "bioasq_train.csv")
test_jsonl_path = os.path.join(output_dir, "bioasq_test.jsonl")
test_csv_path = os.path.join(output_dir, "bioasq_test.csv")

# Define system prompt
system_prompt = "You are a medical and biological expert. Answer the question accurately."


def process_bioasq(data, include_id=True):
    jsonl_data = []
    csv_data = []

    for idx, q in enumerate(data.get("questions", [])):
        if "body" in q and "ideal_answer" in q and len(q["ideal_answer"]) > 0:
            question = q["body"]
            answer = q["ideal_answer"][0]

            # JSONL format
            jsonl_data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

            # CSV format
            row = {
                "question": question,
                "answer": answer
            }
            if include_id:
                row["id"] = idx + 1
            csv_data.append(row)

    return jsonl_data, csv_data


# ----------- Train Set -----------
with open(train_input_path, "r") as f:
    train_data = json.load(f)

train_jsonl, train_csv = process_bioasq(train_data, include_id=True)

# Save JSONL
with open(train_jsonl_path, "w") as f:
    for entry in train_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save CSV
pd.DataFrame(train_csv).to_csv(train_csv_path, index=False)

print(f"Saved {len(train_jsonl)} train entries:")
print(f"- JSONL: {train_jsonl_path}")
print(f"- CSV  : {train_csv_path}")


# ----------- Test Set -----------
with open(test_input_path, "r") as f:
    test_data = json.load(f)

test_jsonl, test_csv = process_bioasq(test_data, include_id=False)

# Save JSONL
with open(test_jsonl_path, "w") as f:
    for entry in test_jsonl:
        json.dump(entry, f)
        f.write("\n")

# Save CSV
pd.DataFrame(test_csv)[["question", "answer"]].to_csv(test_csv_path, index=False)

print(f"Saved {len(test_jsonl)} test entries:")
print(f"- JSONL: {test_jsonl_path}")
print(f"- CSV  : {test_csv_path}")