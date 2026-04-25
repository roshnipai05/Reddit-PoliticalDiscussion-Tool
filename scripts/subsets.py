import json
import subprocess
import sys
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
POST_LIMIT = 5000
POSTS_IN = 'data/cleaned/posts_clean.jsonl'
POSTS_OUT = 'data/cleaned/posts_subset.jsonl'
COMMENTS_IN = 'data/cleaned/comments_clean.jsonl'
COMMENTS_OUT = 'data/cleaned/comments_subset.jsonl'

MODEL_CACHE_DIR = "data/models/huggingface"

# Directory for Topic Modeling results
OUT_DIR_TA = Path('data/topic_analysis')
# Stance Analysis results go to data/topic_stance_preview (sibling of topic_analysis),
# which is exactly where build_app_bundle.py reads from.
OUT_DIR_SA = Path('data/topic_stance_preview')

# Ensure directories exist
OUT_DIR_TA.mkdir(parents=True, exist_ok=True)
OUT_DIR_SA.mkdir(parents=True, exist_ok=True)

post_ids = set()

# 1. Process Posts
print(f"Creating subset: Extracting first {POST_LIMIT} posts...")
with open(POSTS_OUT, 'w', encoding='utf-8') as f_out:
    with open(POSTS_IN, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            if i >= POST_LIMIT: break
            try:
                post = json.loads(line)
                pid = post['post_id']
                post_ids.add(pid)
                f_out.write(line)
            except (json.JSONDecodeError, KeyError):
                continue

# 2. Process Comments
print(f"Filtering comments for {len(post_ids)} posts...")
with open(COMMENTS_OUT, 'w', encoding='utf-8') as f_out:
    with open(COMMENTS_IN, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                c = json.loads(line)
                pid = c.get('link_id', c.get('post_id', ''))
                if pid.startswith('t3_'): pid = pid[3:]
                if pid in post_ids:
                    f_out.write(line)
            except json.JSONDecodeError:
                continue

print("Subset created successfully. Starting Analysis Pipeline...")
print("-" * 30)

# --- PIPELINE EXECUTION ---

# Command A: Topic Modeling
# Outputs: aggregate_stats.json, run_metadata.json, topic_summary.json,
#           topic_summary.csv, post_topics.csv -- all under OUT_DIR_TA.
# NOTE: topic_modeling_analysis.py writes topic_summary.json natively;
#       no CSV->JSON conversion step is needed.
print(f"\nRunning Step A: Topic Modeling\n")
cmd_a = [
    sys.executable, "scripts/topic_modeling_analysis.py",
    "--posts", POSTS_OUT,
    "--out-dir", str(OUT_DIR_TA),
    "--model-cache-dir", MODEL_CACHE_DIR,
    "--target-topics", "15",
    "--min-topic-size", "20",
    "--reduce-outliers",
]
result_a = subprocess.run(cmd_a)
if result_a.returncode != 0:
    print("[!] Topic Modeling failed. Stopping.")
    sys.exit(1)


# Command B: Stance Analysis
# Reads from OUT_DIR_TA; outputs to OUT_DIR_SA so it does not
# overwrite run_metadata.json or other TA files.
print(f"\nRunning Step B: Stance Analysis\n")
cmd_b = [
    sys.executable, "scripts/topic_stance_analysis.py",
    "--comments", COMMENTS_OUT,
    "--post-topics", str(OUT_DIR_TA / "post_topics.csv"),      # Input from TA
    "--topic-summary", str(OUT_DIR_TA / "topic_summary.json"), # Input from TA
    "--out-dir", str(OUT_DIR_SA),                              # Output to its own folder
    "--model-cache-dir", MODEL_CACHE_DIR,
]
result_b = subprocess.run(cmd_b)
if result_b.returncode != 0:
    print("[!] Stance Analysis failed. Stopping.")
    sys.exit(1)


# Command C: Build App Bundle
print(f"\nRunning Step C: Building App Bundle\n")
cmd_c = [sys.executable, "scripts/build_app_bundle.py"]
result_c = subprocess.run(cmd_c)

if result_c.returncode == 0:
    print("\n" + "="*30)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("You can now refresh your localhost app.")
    print("="*30)
else:
    print("[!] Bundling failed.")