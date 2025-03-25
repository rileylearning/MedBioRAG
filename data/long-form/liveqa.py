import os
import json
import pandas as pd

# Paths
train1_path = os.path.join("data", "liveqa", "train1.jsonl")
train2_path = os.path.join("data", "liveqa", "train2.jsonl")
test_path = os.path.join("data", "liveqa", "test.jsonl")

output_dir = os.path.join("data", "long-form")
os.makedirs(output_dir, exist_ok=True)

train_jsonl_path = os.path.join(output_dir, "liveqa_train.jsonl")
train_csv_path = os.path.join(output_dir, "liveqa_train.csv")
test_jsonl_path = os.path.join(output_dir, "liveqa_test.jsonl")
test_csv_path = os.path.join(output_dir, "liveqa_test.csv")

system_prompt = "You are a multilingual medical assistant. Provide accurate explanations for medical questions in the detected language."


def load_train_file(path):
    results = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data.get("NLM_SUMMARY", "")
            answer = data.get("ANSWER", "")
            if question and answer:
                results.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                })
    return results


def load_test_file(path):
    results = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data.get("NLM_SUMMARY", "")
            answers = data.get("REFERENCE_ANSWERS", [])
            answer_texts = [ans.get("ANSWER", "") for ans in answers if "ANSWER" in ans]
            answer = "\n".join(filter(None, answer_texts))
            if question and answer:
                results.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                })
    return results


# -------- Load & save train data --------
train_data_1 = load_train_file(train1_path)
train_data_2 = load_train_file(train2_path)
train_data = train_data_1 + train_data_2

# Save JSONL
with open(train_jsonl_path, 'w') as f:
    for item in train_data:
        json.dump(item, f)
        f.write("\n")

# Save CSV
train_rows = []
for i, item in enumerate(train_data):
    user = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
    assistant = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
    train_rows.append({
        "id": i + 1,
        "question": user,
        "answer": assistant
    })
pd.DataFrame(train_rows).to_csv(train_csv_path, index=False)


# -------- Load & save test data --------
test_data = load_test_file(test_path)

# Save JSONL
with open(test_jsonl_path, 'w') as f:
    for item in test_data:
        json.dump(item, f)
        f.write("\n")

# Save CSV
test_rows = []
for item in test_data:
    user = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
    assistant = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
    test_rows.append({
        "question": user,
        "answer": assistant
    })
pd.DataFrame(test_rows).to_csv(test_csv_path, index=False)


# -------- Summary --------
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Saved to:")
print(f"- {train_jsonl_path}")
print(f"- {train_csv_path}")
print(f"- {test_jsonl_path}")
print(f"- {test_csv_path}")