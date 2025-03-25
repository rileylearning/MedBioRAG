import os
import json
import pandas as pd

# Base paths
train_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "BioASQ-training12b", "training12b_new.json")
test_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "Task12BGoldenEnriched", "12B4_golden.json")
output_dir = os.path.join("data", "close-ended", "bioasq")
os.makedirs(output_dir, exist_ok=True)

# Output files
train_jsonl_path = os.path.join(output_dir, "bioasq_train.jsonl")
train_csv_path = os.path.join(output_dir, "bioasq_train.csv")
test_jsonl_path = os.path.join(output_dir, "bioasq_test.jsonl")
test_csv_path = os.path.join(output_dir, "bioasq_test.csv")

# Prompt for both sets
system_prompt = "You are a multilingual medical assistant. Answer only with 'yes' or 'no'.  Do not add anything else."


def process_questions(data, include_context=False):
    jsonl_data = []
    csv_data = []

    for idx, question in enumerate(data.get("questions", [])):
        answer = question.get("exact_answer")
        if answer not in ["yes", "no"]:
            continue

        # Create context if required
        if include_context:
            snippets = question.get("snippets", [])
            context = " ".join(snippet.get("text", "") for snippet in snippets)
            user_content = f"Question: {question.get('body')}\nContext: {context}"
        else:
            user_content = question.get("body")

        # JSONL format
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]
        })

        # CSV format
        csv_data.append({
            "id": idx + 1 if not include_context else None,
            "question": question.get("body"),
            "answer": answer
        })

    return jsonl_data, csv_data


# ----------- Train -----------
with open(train_input_path, 'r') as f:
    train_data = json.load(f)

train_jsonl, train_csv_rows = process_questions(train_data, include_context=False)

# Save train.jsonl
with open(train_jsonl_path, 'w') as f:
    for entry in train_jsonl:
        json.dump(entry, f)
        f.write('\n')

# Save train.csv with ID
train_df = pd.DataFrame(train_csv_rows)
train_df.to_csv(train_csv_path, index=False)

print(f"Saved {len(train_jsonl)} train entries to:")
print(f"- {train_jsonl_path}")
print(f"- {train_csv_path}")


# ----------- Test -----------
with open(test_input_path, 'r') as f:
    test_data = json.load(f)

test_jsonl, test_csv_rows = process_questions(test_data, include_context=True)

# Save test.jsonl
with open(test_jsonl_path, 'w') as f:
    for entry in test_jsonl:
        json.dump(entry, f)
        f.write('\n')

# Save test.csv without ID
test_df = pd.DataFrame(test_csv_rows)
test_df = test_df[["question", "answer"]]
test_df.to_csv(test_csv_path, index=False)

print(f"Saved {len(test_jsonl)} test entries to:")
print(f"- {test_jsonl_path}")
print(f"- {test_csv_path}")