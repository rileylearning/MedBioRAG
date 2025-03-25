import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Base directory for TREC-COVID raw files
base_dir = "data/trec-covid"
output_dir = "data/retrieval"
os.makedirs(output_dir, exist_ok=True)

# File paths
metadata_path = os.path.join(base_dir, "metadata.csv")
docids_path = os.path.join(base_dir, "docids-rnd5.txt")
topics_path = os.path.join(base_dir, "topics-rnd5.xml")
qrels_path = os.path.join(base_dir, "qrels-covid_d5_j0.5-5.txt")

# Output paths
corpus_jsonl_path = os.path.join(output_dir, "corpus.jsonl")
split_corpus_dir = os.path.join(output_dir, "split_corpus")
topics_csv_path = os.path.join(output_dir, "topics-rnd5.csv")
filtered_qrels_path = os.path.join(output_dir, "qrels-covid_d5_j0.5-5_filtered.csv")

os.makedirs(split_corpus_dir, exist_ok=True)

# Load document IDs from docids-rnd5.txt
with open(docids_path, "r") as f:
    doc_ids = set(line.strip() for line in f.readlines())

# Load metadata.csv and filter relevant documents
df_metadata = pd.read_csv(metadata_path)
df_filtered = df_metadata[df_metadata["cord_uid"].isin(doc_ids)][
    ["cord_uid", "title", "abstract", "pdf_json_files", "pmc_json_files", "s2_id"]
]

# Extract paper URL from available fields
def get_paper_url(row):
    for col in ["pmc_json_files", "pdf_json_files"]:
        if pd.notna(row[col]):
            article_id = row[col].split(";")[0].split("/")[-1].replace(".json", "").replace(".xml", "")
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/"
    if pd.notna(row["s2_id"]):
        return f"https://www.semanticscholar.org/paper/{row['s2_id']}"
    return ""

# Add URL column
df_filtered["url"] = df_filtered.apply(get_paper_url, axis=1)

# Save corpus as JSONL file
with open(corpus_jsonl_path, "w", encoding="utf-8") as f:
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Writing corpus.jsonl"):
        json.dump({
            "_id": row["cord_uid"],
            "title": row["title"] or "Unknown",
            "text": row["abstract"] or "No abstract available",
            "metadata": {"url": row["url"]}
        }, f)
        f.write("\n")

# Split corpus into smaller CSV chunks
chunk_size = 50000
df_final = df_filtered.rename(columns={"cord_uid": "id"})
df_final["title"] = df_final["title"].fillna("") + " " + df_final["abstract"].fillna("")
df_final = df_final[["id", "title"]]

for i, start in enumerate(range(0, len(df_final), chunk_size)):
    part_df = df_final.iloc[start:start + chunk_size]
    part_path = os.path.join(split_corpus_dir, f"trec-covid_corpus_part_{i + 1}.csv")
    part_df.to_csv(part_path, index=False)

# Convert topics-rnd5.xml to CSV format
tree = ET.parse(topics_path)
root = tree.getroot()
topics = []
for topic in root.findall("topic"):
    topics.append({
        "query-id": topic.get("number"),
        "query": topic.findtext("query", default=""),
        "question": topic.findtext("question", default=""),
        "narrative": topic.findtext("narrative", default="")
    })
pd.DataFrame(topics).to_csv(topics_csv_path, index=False)

# Parse and reformat qrels file with standardized column names
df_qrels = pd.read_csv(qrels_path, sep=" ", names=["query_id", "iteration", "doc_id", "relevance"])
df_qrels.drop(columns=["iteration"], inplace=True)
df_qrels.to_csv(filtered_qrels_path, index=False)

print("All files successfully created:")
print("-", corpus_jsonl_path)
print("-", topics_csv_path)
print("-", filtered_qrels_path)
print("- Split CSVs in:", split_corpus_dir)