"""Evaluate the RAG system against the ground-truth QA set.

Reads:
    scripts/eval_set.json                 (ground-truth QA pairs)
    data/chroma_db/                       (built by build_index.py)

Writes:
    outputs/eval_raw_answers.json         (all questions + model answers)
    outputs/eval_scores.csv               (ROUGE-L + BERTScore per question per model)
    outputs/eval_manual_template.csv      (pre-filled CSV for manual faithfulness/relevance flags)
    outputs/evaluation_report.md          (final report, generated after manual flags are filled)

Usage (two-step process):
    # Step 1: run pipeline, compute automatic metrics
    python scripts/evaluate_rag.py

    # Step 2: open outputs/eval_manual_template.csv, fill in the 0/1 flags,
    #         save as outputs/eval_manual_filled.csv, then generate report:
    python scripts/evaluate_rag.py --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = ROOT / "scripts" / "eval_set.json"
CHROMA_DIR = ROOT / "data" / "chroma_db"
MODEL_CACHE_DIR = ROOT / "data" / "models" / "huggingface"
OUTPUTS_DIR = ROOT / "outputs"

RAW_ANSWERS_PATH = OUTPUTS_DIR / "eval_raw_answers.json"
SCORES_CSV_PATH = OUTPUTS_DIR / "eval_scores.csv"
MANUAL_TEMPLATE_PATH = OUTPUTS_DIR / "eval_manual_template.csv"
MANUAL_FILLED_PATH = OUTPUTS_DIR / "eval_manual_filled.csv"
REPORT_PATH = OUTPUTS_DIR / "evaluation_report.md"

# Groq free tier: ~6000 tokens/min. With ~15k token prompts, a short sleep
# between calls prevents hitting the per-minute limit during sequential fallback.
GROQ_INTER_CALL_SLEEP = 2.0   # seconds between sequential Groq calls


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip pipeline run; load existing raw answers and generate report only. "
             "Requires outputs/eval_manual_filled.csv to exist.",
    )
    parser.add_argument(
        "--eval-set", default=str(EVAL_SET_PATH),
        help="Path to eval_set.json",
    )
    parser.add_argument("--chroma-dir", default=str(CHROMA_DIR))
    parser.add_argument("--model-cache-dir", default=str(MODEL_CACHE_DIR))
    parser.add_argument(
        "--skip-groq", action="store_true",
        help="Skip Groq calls (useful for testing with only Gemini)",
    )
    parser.add_argument(
        "--skip-gemini", action="store_true",
        help="Skip Gemini calls",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rouge_l(reference: str, generated: str) -> float:
    """Compute ROUGE-L F1 between reference and generated answer."""
    from rouge_score import rouge_scorer  # type: ignore
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, generated)
    return round(float(result["rougeL"].fmeasure), 4)


def compute_bertscore(references: list[str], generateds: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for a batch of (reference, generated) pairs.
    Uses roberta-large backbone for a balance of quality and speed.
    Returns list of F1 scores (floats).
    """
    from bert_score import score as bert_score  # type: ignore

    _, _, F1 = bert_score(
        generateds,
        references,
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    return [round(float(f), 4) for f in F1.tolist()]


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def load_eval_set(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


async def run_all_questions(
    eval_set: list[dict[str, Any]],
    post_collection,
    comment_collection,
    model_cache_dir: str,
    skip_groq: bool,
    skip_gemini: bool,
) -> list[dict[str, Any]]:
    """
    Run every question through the RAG pipeline.
    Calls Groq and Gemini in parallel per question.
    Returns list of result dicts.
    """
    # Import here to avoid circular dependency (rag_query imports nothing from here)
    from rag_query import retrieve, format_context, build_prompt, query_both_async
    from rag_query import query_groq, query_gemini

    results = []
    total = len(eval_set)

    for i, item in enumerate(eval_set, start=1):
        qid = item["id"]
        question = item["question"]
        qtype = item["type"]
        answerable = item["answerable"]

        print(f"[{i}/{total}] {qid} ({qtype}) — {question[:70]}...")

        retrieval = retrieve(question, post_collection, comment_collection, model_cache_dir)
        context = format_context(retrieval["posts"], retrieval["comments_by_post"])
        system, user = build_prompt(question, context, retrieval["no_answer_flag"])

        groq_answer = None
        gemini_answer = None

        if not skip_groq and not skip_gemini:
            answers = await query_both_async(system, user)
            groq_answer = answers["groq"]
            gemini_answer = answers["gemini"]
        elif not skip_groq:
            try:
                groq_answer = query_groq(system, user)
            except Exception as exc:
                groq_answer = f"[error: {exc}]"
            time.sleep(GROQ_INTER_CALL_SLEEP)
        elif not skip_gemini:
            try:
                gemini_answer = query_gemini(system, user)
            except Exception as exc:
                gemini_answer = f"[error: {exc}]"

        results.append({
            "id": qid,
            "type": qtype,
            "answerable": answerable,
            "question": question,
            "reference_answer": item["reference_answer"],
            "no_answer_flag": retrieval["no_answer_flag"],
            "max_cosine_sim": round(retrieval["max_cosine_sim"], 4),
            "retrieved_post_ids": [p["post_id"] for p in retrieval["posts"]],
            "groq_answer": groq_answer or "",
            "gemini_answer": gemini_answer or "",
        })

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compute ROUGE-L and BERTScore for every (model, question) pair.
    Returns a DataFrame with one row per question.
    """
    references = [r["reference_answer"] for r in results]
    groq_answers = [r["groq_answer"] for r in results]
    gemini_answers = [r["gemini_answer"] for r in results]

    print("Computing ROUGE-L scores...")
    groq_rouge = [compute_rouge_l(ref, gen) for ref, gen in zip(references, groq_answers)]
    gemini_rouge = [compute_rouge_l(ref, gen) for ref, gen in zip(references, gemini_answers)]

    print("Computing BERTScore (this may take a minute)...")
    # Filter out empty answers to avoid BERTScore errors, then re-insert zeros
    def safe_bertscore(refs, gens):
        scores = []
        for ref, gen in zip(refs, gens):
            if not gen.strip():
                scores.append(0.0)
            else:
                scores.extend(compute_bertscore([ref], [gen]))
        return scores

    groq_bert = safe_bertscore(references, groq_answers)
    gemini_bert = safe_bertscore(references, gemini_answers)

    rows = []
    for r, gr_rouge, gm_rouge, gr_bert, gm_bert in zip(
        results, groq_rouge, gemini_rouge, groq_bert, gemini_bert
    ):
        rows.append({
            "id": r["id"],
            "type": r["type"],
            "answerable": r["answerable"],
            "question": r["question"][:80],
            "no_answer_flag": r["no_answer_flag"],
            "max_cosine_sim": r["max_cosine_sim"],
            "groq_rouge_l": gr_rouge,
            "gemini_rouge_l": gm_rouge,
            "groq_bertscore_f1": gr_bert,
            "gemini_bertscore_f1": gm_bert,
            # Manual columns — left blank for human annotation
            "groq_faithful": "",
            "gemini_faithful": "",
            "groq_relevant": "",
            "gemini_relevant": "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    scores_df: pd.DataFrame,
    raw_results: list[dict[str, Any]],
) -> str:
    """
    Generate the full evaluation_report.md.
    If manual columns are filled (non-empty), include faithfulness/relevance stats.
    """
    has_manual = scores_df["groq_faithful"].astype(str).str.strip().ne("").any()

    lines = [
        "# RAG System Evaluation Report",
        "",
        "## 1. Configuration",
        "",
        f"- **Embedding model**: sentence-transformers/all-mpnet-base-v2",
        f"- **Vector store**: ChromaDB (two collections: reddit_posts, reddit_comments)",
        f"- **Retrieval**: top-5 posts (diversity-filtered) + up to 5 comments per post",
        f"- **Re-ranking**: cosine similarity × log(1 + reddit_score)",
        f"- **No-answer threshold**: cosine similarity < 0.35",
        f"- **LLM A (Groq)**: llama-3.3-70b-versatile",
        f"- **LLM B (Gemini)**: gemini-2.0-flash",
        f"- **Evaluation set**: {len(scores_df)} questions "
        f"({scores_df['type'].value_counts().to_dict()})",
        "",
        "---",
        "",
        "## 2. Results Table",
        "",
    ]

    # Build results table header
    if has_manual:
        header = (
            "| ID | Type | Groq ROUGE-L | Gemini ROUGE-L | "
            "Groq BERTScore | Gemini BERTScore | "
            "Groq Faithful | Gemini Faithful | "
            "Groq Relevant | Gemini Relevant |"
        )
        sep = "|" + "|".join(["-" * 16] * 10) + "|"
    else:
        header = (
            "| ID | Type | Groq ROUGE-L | Gemini ROUGE-L | "
            "Groq BERTScore | Gemini BERTScore |"
        )
        sep = "|" + "|".join(["-" * 16] * 6) + "|"

    lines += [header, sep]

    for _, row in scores_df.iterrows():
        if has_manual:
            line = (
                f"| {row['id']} | {row['type']} "
                f"| {row['groq_rouge_l']:.4f} | {row['gemini_rouge_l']:.4f} "
                f"| {row['groq_bertscore_f1']:.4f} | {row['gemini_bertscore_f1']:.4f} "
                f"| {row['groq_faithful']} | {row['gemini_faithful']} "
                f"| {row['groq_relevant']} | {row['gemini_relevant']} |"
            )
        else:
            line = (
                f"| {row['id']} | {row['type']} "
                f"| {row['groq_rouge_l']:.4f} | {row['gemini_rouge_l']:.4f} "
                f"| {row['groq_bertscore_f1']:.4f} | {row['gemini_bertscore_f1']:.4f} |"
            )
        lines.append(line)

    # Summary row
    mean_cols = ["groq_rouge_l", "gemini_rouge_l", "groq_bertscore_f1", "gemini_bertscore_f1"]
    means = scores_df[mean_cols].mean()
    if has_manual:
        def pct(col):
            vals = pd.to_numeric(scores_df[col], errors="coerce")
            return f"{vals.mean() * 100:.1f}%" if vals.notna().any() else "N/A"

        mean_row = (
            f"| **Mean** | — "
            f"| **{means['groq_rouge_l']:.4f}** | **{means['gemini_rouge_l']:.4f}** "
            f"| **{means['groq_bertscore_f1']:.4f}** | **{means['gemini_bertscore_f1']:.4f}** "
            f"| **{pct('groq_faithful')}** | **{pct('gemini_faithful')}** "
            f"| **{pct('groq_relevant')}** | **{pct('gemini_relevant')}** |"
        )
    else:
        mean_row = (
            f"| **Mean** | — "
            f"| **{means['groq_rouge_l']:.4f}** | **{means['gemini_rouge_l']:.4f}** "
            f"| **{means['groq_bertscore_f1']:.4f}** | **{means['gemini_bertscore_f1']:.4f}** |"
        )
    lines.append(mean_row)
    lines.append("")

    # Adversarial sub-table
    adversarial = scores_df[scores_df["answerable"] == False]  # noqa: E712
    if not adversarial.empty:
        lines += [
            "---",
            "",
            "## 3. Adversarial Question Behaviour",
            "",
            "| ID | Question (truncated) | no_answer_flag | "
            "Groq Refused Correctly | Gemini Refused Correctly |",
            "|---|---|---|---|---|",
        ]
        raw_lookup = {r["id"]: r for r in raw_results}
        for _, row in adversarial.iterrows():
            raw = raw_lookup.get(row["id"], {})
            question_short = row["question"][:60]
            flag = "✓" if row["no_answer_flag"] else "✗"

            groq_ans = raw.get("groq_answer", "")
            gemini_ans = raw.get("gemini_answer", "")
            refusal_phrase = "does not contain sufficient information"
            groq_refused = "✓" if refusal_phrase.lower() in groq_ans.lower() else "✗ (check manually)"
            gemini_refused = "✓" if refusal_phrase.lower() in gemini_ans.lower() else "✗ (check manually)"

            lines.append(
                f"| {row['id']} | {question_short}... | {flag} "
                f"| {groq_refused} | {gemini_refused} |"
            )
        lines.append("")

    # Per-type breakdown
    lines += [
        "---",
        "",
        "## 4. Performance by Question Type",
        "",
        "| Type | Count | Groq ROUGE-L | Gemini ROUGE-L | "
        "Groq BERTScore | Gemini BERTScore |",
        "|---|---|---|---|---|---|",
    ]
    for qtype, group in scores_df.groupby("type"):
        lines.append(
            f"| {qtype} | {len(group)} "
            f"| {group['groq_rouge_l'].mean():.4f} | {group['gemini_rouge_l'].mean():.4f} "
            f"| {group['groq_bertscore_f1'].mean():.4f} | {group['gemini_bertscore_f1'].mean():.4f} |"
        )
    lines.append("")

    # Model answers for reference
    lines += [
        "---",
        "",
        "## 5. Full Model Answers",
        "",
        "_Generated answers for manual review._",
        "",
    ]
    for r in raw_results:
        lines += [
            f"### {r['id']} — {r['type']} {'(adversarial)' if not r['answerable'] else ''}",
            f"**Question**: {r['question']}",
            f"**Reference**: {r['reference_answer']}",
            f"**no_answer_flag**: {r['no_answer_flag']} | "
            f"**max_cosine_sim**: {r['max_cosine_sim']}",
            "",
            f"**Groq answer**:",
            f"> {r['groq_answer']}",
            "",
            f"**Gemini answer**:",
            f"> {r['gemini_answer']}",
            "",
            "---",
            "",
        ]

    # Qualitative analysis placeholder
    lines += [
        "## 6. Qualitative Analysis",
        "",
        "_(Fill in after reviewing the full answers table above.)_",
        "",
        "### Where Groq succeeds",
        "",
        "- ",
        "",
        "### Where Groq fails",
        "",
        "- ",
        "",
        "### Where Gemini succeeds",
        "",
        "- ",
        "",
        "### Where Gemini fails",
        "",
        "- ",
        "",
        "### Retrieval quality observations",
        "",
        "- ",
        "",
        "### Adversarial behaviour",
        "",
        "- ",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_set = load_eval_set(Path(args.eval_set))

    # --- Report-only mode ---
    if args.report_only:
        if not RAW_ANSWERS_PATH.exists():
            raise FileNotFoundError(
                f"Raw answers not found at {RAW_ANSWERS_PATH}. "
                "Run without --report-only first."
            )
        raw_results = json.loads(RAW_ANSWERS_PATH.read_text(encoding="utf-8"))

        # Use filled manual CSV if available, otherwise fall back to scores CSV
        if MANUAL_FILLED_PATH.exists():
            scores_df = pd.read_csv(MANUAL_FILLED_PATH)
            print(f"Loaded manual flags from {MANUAL_FILLED_PATH}")
        elif SCORES_CSV_PATH.exists():
            scores_df = pd.read_csv(SCORES_CSV_PATH)
            print(f"No manual CSV found; using {SCORES_CSV_PATH} (no faithfulness stats)")
        else:
            raise FileNotFoundError(
                f"No scores CSV found. Run without --report-only first."
            )

        report = generate_report(scores_df, raw_results)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"Report written: {REPORT_PATH}")
        return

    # --- Full pipeline run ---
    # Import rag_query components
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from rag_query import get_collections  # noqa: E402

    post_collection, comment_collection = get_collections(args.chroma_dir)

    print(f"Running {len(eval_set)} questions through RAG pipeline...")
    raw_results = asyncio.run(
        run_all_questions(
            eval_set=eval_set,
            post_collection=post_collection,
            comment_collection=comment_collection,
            model_cache_dir=args.model_cache_dir,
            skip_groq=args.skip_groq,
            skip_gemini=args.skip_gemini,
        )
    )

    # Save raw answers
    RAW_ANSWERS_PATH.write_text(
        json.dumps(raw_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Raw answers saved: {RAW_ANSWERS_PATH}")

    # Compute automatic metrics
    scores_df = score_results(raw_results)
    scores_df.to_csv(SCORES_CSV_PATH, index=False)
    print(f"Scores CSV saved: {SCORES_CSV_PATH}")

    # Save manual template (identical structure, manual columns blank for annotation)
    scores_df.to_csv(MANUAL_TEMPLATE_PATH, index=False)
    print(f"\nManual annotation template saved: {MANUAL_TEMPLATE_PATH}")
    print(
        "Next step: open that CSV, fill in groq_faithful, gemini_faithful, "
        "groq_relevant, gemini_relevant columns with 0 or 1,\n"
        f"save as {MANUAL_FILLED_PATH}, then run:\n"
        "    python scripts/evaluate_rag.py --report-only"
    )

    # Generate preliminary report (no manual flags yet)
    report = generate_report(scores_df, raw_results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Preliminary report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()