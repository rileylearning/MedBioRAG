import os
import json
import pandas as pd

# Set input and output paths
input_base_path = os.path.join("data", "medqa", "questions", "US")
output_base_path = os.path.join("data", "close-ended", "medqa")

# Ensure output directory exists
os.makedirs(output_base_path, exist_ok=True)

# Filenames
input_files = {
    "train": "train.jsonl",
    "val": "dev.jsonl",
    "test": "test.jsonl"
}
output_jsonl = {
    "train": "medqa_train.jsonl",
    "val": "medqa_val.jsonl",
    "test": "medqa_test.jsonl"
}
output_csv_test = "medqa_test.csv"

# Prompt template
system_prompt = "You are taking the United States Medical Licensing Examination (USMLE). Respond with only one option (A, B, C, D, or E) based on the question and options provided."

# Process each split
for split, filename in input_files.items():
    input_path = os.path.join(input_base_path, filename)
    jsonl_output_path = os.path.join(output_base_path, output_jsonl[split])

    converted_data = []
    csv_rows = []

    with open(input_path, "r") as infile:
        for line in infile:
            entry = json.loads(line.strip())

            question = entry["question"]
            options = entry["options"]
            answer = entry["answer_idx"]

            user_content = f"{question}\nOptions:"
            for key, value in options.items():
                user_content += f"\n{key}) {value}"

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ]
            }

            converted_data.append(conversation)

            # For CSV output (test only)
            if split == "test":
                csv_rows.append({
                    "question": question,
                    "answer": answer
                })

    # Save JSONL output
    with open(jsonl_output_path, "w") as outfile:
        for conversation in converted_data:
            json.dump(conversation, outfile)
            outfile.write("\n")
    print(f"Saved {len(converted_data)} entries to {jsonl_output_path}")

    # Save CSV for test set
    if split == "test":
        df = pd.DataFrame(csv_rows)
        csv_output_path = os.path.join(output_base_path, output_csv_test)
        df.to_csv(csv_output_path, index=False)
        print(f"Saved test CSV to {csv_output_path}")