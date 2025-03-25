import os
import json
import random
import pandas as pd

# Input Excel file path
input_excel_path = os.path.join("data", "medicationqa", "MedInfo2019-QA-Medications.xlsx")

# Output directory and files
output_dir = os.path.join("data", "long-form")
os.makedirs(output_dir, exist_ok=True)

jsonl_path = os.path.join(output_dir, "medicationqa_all.jsonl")
train_jsonl_path = os.path.join(output_dir, "medicationqa_train.jsonl")
test_jsonl_path = os.path.join(output_dir, "medicationqa_test.jsonl")
train_csv_path = os.path.join(output_dir, "medicationqa_train.csv")
test_csv_path = os.path.join(output_dir, "medicationqa_test.csv")

# Prompt template
system_prompt = "You are a medical and biological expert. Answer the question accurately."

# Step 1: Load the Excel file and build conversation format
df = pd.read_excel(input_excel_path)

conversations = []
for _, row in df.iterrows():
    question = row.get("Question")
    answer = row.get("Answer")

    if pd.notna(question) and pd.notna(answer):
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        conversations.append(conversation)

# Step 2: Save all data to JSONL
with open(jsonl_path, 'w') as f:
    for entry in conversations:
        json.dump(entry, f)
        f.write("\n")

# Step 3: Split into train/test (90% train, 10% test)
random.shuffle(conversations)
split_index = int(len(conversations) * 0.9)
train_set = conversations[:split_index]
test_set = conversations[split_index:]

# Save train/test JSONL
with open(train_jsonl_path, 'w') as f:
    for entry in train_set:
        json.dump(entry, f)
        f.write("\n")

with open(test_jsonl_path, 'w') as f:
    for entry in test_set:
        json.dump(entry, f)
        f.write("\n")

# Step 4: Generate train CSV with ID
train_rows = []
for i, convo in enumerate(train_set):
    user_msg = next((m["content"] for m in convo["messages"] if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in convo["messages"] if m["role"] == "assistant"), "")
    train_rows.append({
        "id": i + 1,
        "question": user_msg,
        "answer": assistant_msg
    })
pd.DataFrame(train_rows).to_csv(train_csv_path, index=False)

# Step 5: Generate test CSV without ID
test_rows = []
for convo in test_set:
    user_msg = next((m["content"] for m in convo["messages"] if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in convo["messages"] if m["role"] == "assistant"), "")
    test_rows.append({
        "question": user_msg,
        "answer": assistant_msg
    })
pd.DataFrame(test_rows).to_csv(test_csv_path, index=False)

# Print summary
print(f"Total entries: {len(conversations)}")
print(f"Train set: {len(train_set)}")
print(f"Test set: {len(test_set)}")
print(f"Files saved:")
print(f"- {train_jsonl_path}")
print(f"- {test_jsonl_path}")
print(f"- {train_csv_path}")
print(f"- {test_csv_path}")