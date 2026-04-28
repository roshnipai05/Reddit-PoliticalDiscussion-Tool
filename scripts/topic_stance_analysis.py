"""Stance analysis for Reddit topic discussions.

Targets Project Part 1 task 1.4:
- classify each comment as broadly support or opposing the dominant position
- group users by stance
- summarize key arguments made by each side

The implementation is intentionally application-oriented:
- writes per-comment stance labels for UI drill-down
- writes per-topic summaries, dominant-position metadata, and user groups
- uses dense sentence embeddings plus two-cluster stance grouping

This is a broad stance proxy, not gold-label NLI. For each topic, comments are
clustered into two discourse camps using semantic embeddings. The larger camp
is treated as the dominant/support side and the smaller camp as the opposing
side. Topic summaries are extractive and based on representative comments and
cluster-specific keywords.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_summaries import groq_available, groq_json_completion


DEFAULT_SEED = 42
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MIN_COMMENTS_PER_TOPIC = 60
MAX_USER_PREVIEW = 50
MAX_REPRESENTATIVE_COMMENTS = 8
ROOT = Path(__file__).resolve().parents[1]
STANCE_STOPWORDS = {
    "actually",
    "american",
    "americans",
    "argument",
    "arguments",
    "candidate",
    "country",
    "debate",
    "democrat",
    "democrats",
    "election",
    "elections",
    "government",
    "issue",
    "issues",
    "just",
    "like",
    "really",
    "know",
    "think",
    "people",
    "person",
    "good",
    "bad",
    "thing",
    "things",
    "does",
    "did",
    "don",
    "isn",
    "wasn",
    "would",
    "could",
    "should",
    "also",
    "because",
    "policy",
    "political",
    "politics",
    "trump",
    "biden",
    "harris",
    "voters",
    "year",
    "years",
    "going",
    "say",
    "says",
    "said",
    "state",
    "states",
    "president",
    "time",
    "vote",
    "votes",
    "voting",
    "campaign",
    "need",
    "way",
    "things",
    "point",
    "kind",
    "lot",
    "really",
}
SUPPORT_CUES = {
    "agree",
    "exactly",
    "same",
    "yes",
    "absolutely",
    "definitely",
    "valid",
    "makes sense",
    "right",
    "true",
    "support",
    "fair",
}
OPPOSE_CUES = {
    "disagree",
    "but",
    "however",
    "wrong",
    "no",
    "not true",
    "unfair",
    "instead",
    "actually",
    "counterpoint",
    "though",
    "yet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comments", default=None)
    parser.add_argument("--post-topics", default=None)
    parser.add_argument("--topic-summary", default=None)
    parser.add_argument("--out-dir", default="data/topic_stance")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--model-cache-dir", default="data/models/huggingface")
    parser.add_argument("--min-comments-per-topic", type=int, default=MIN_COMMENTS_PER_TOPIC)
    parser.add_argument("--disable-hf-ssl-verify", action="store_true")
    parser.add_argument("--disable-groq-summaries", action="store_true")
    parser.add_argument(
        "--topic-ids",
        default="",
        help="Optional comma-separated topic ids to analyze.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=0,
        help="If set, analyze only the top N topics by matched comment count.",
    )
    parser.add_argument(
        "--max-comments-per-topic",
        type=int,
        default=0,
        help="If set, sample at most this many comments per topic.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("top", "recent", "random"),
        default="top",
        help="How to sample comments when --max-comments-per-topic is set.",
    )
    return parser.parse_args()


def resolve_input_path(provided: str | None, relative_default: str) -> Path:
    if provided:
        return Path(provided)
    path = ROOT / relative_default
    if path.exists():
        return path
    raise FileNotFoundError(f"Input not found at {path}. Pass the path explicitly.")


def configure_huggingface_env(model_cache_dir: Path, disable_ssl_verify: bool) -> None:
    os.environ.setdefault("HF_HOME", str(model_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache_dir))
    if disable_ssl_verify:
        import httpx
        from huggingface_hub import set_client_factory

        def client_factory() -> httpx.Client:
            return httpx.Client(verify=False, follow_redirects=True, timeout=60.0)

        set_client_factory(client_factory)


def load_sentence_model(model_name: str, cache_dir: Path):
    from sentence_transformers import SentenceTransformer

    snapshot_root = cache_dir / f"models--sentence-transformers--{model_name.replace('/', '--')}"
    ref_path = snapshot_root / "refs" / "main"
    if ref_path.exists():
        snapshot_hash = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = snapshot_root / "snapshots" / snapshot_hash
        if snapshot_path.exists():
            return SentenceTransformer(str(snapshot_path), local_files_only=True)

    return SentenceTransformer(model_name, cache_folder=str(cache_dir), local_files_only=True)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_topic_summary(path: Path) -> pd.DataFrame:
    topic_rows = json.loads(path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(topic_rows)
    summary = summary[summary["topic_id"] != -1].copy()
    return summary


def load_post_topics(path: Path, valid_topic_ids: set[int]) -> pd.DataFrame:
    post_topics = pd.read_csv(path)
    post_topics = post_topics[post_topics["topic_id"].isin(valid_topic_ids)].copy()
    return post_topics


def build_comment_frame(comments_path: Path, post_topics: pd.DataFrame) -> pd.DataFrame:
    topic_lookup = {
        row.post_id: {
            "topic_id": int(row.topic_id),
            "topic_created_month": row.created_month,
            "topic_post_title": row.title,
            "topic_post_score": int(row.score),
            "topic_post_num_comments": int(row.num_comments),
            "topic_post_permalink": row.permalink,
            "link_flair_text": row.link_flair_text,
        }
        for row in post_topics.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    for comment in read_jsonl(comments_path):
        post_meta = topic_lookup.get(comment["post_id"])
        if not post_meta:
            continue
        records.append(
            {
                "comment_id": comment["comment_id"],
                "post_id": comment["post_id"],
                "parent_id": comment["parent_id"],
                "created_utc": comment["created_utc"],
                "created_iso": comment["created_iso"],
                "author_hash": comment["author_hash"],
                "body": comment["body"],
                "score": int(comment.get("score") or 0),
                "permalink": comment["permalink"],
                **post_meta,
            }
        )
    if not records:
        raise ValueError("No comments matched the topic-assigned posts.")
    return pd.DataFrame(records)


def parse_topic_ids(raw_value: str) -> set[int]:
    if not raw_value.strip():
        return set()
    return {int(part.strip()) for part in raw_value.split(",") if part.strip()}


def sample_topic_comments(
    topic_comments: pd.DataFrame,
    max_comments_per_topic: int,
    sample_mode: str,
    seed: int,
) -> pd.DataFrame:
    if max_comments_per_topic <= 0 or len(topic_comments) <= max_comments_per_topic:
        return topic_comments.copy()
    if sample_mode == "recent":
        return topic_comments.sort_values(["created_utc", "score"], ascending=[False, False]).head(max_comments_per_topic).copy()
    if sample_mode == "random":
        return topic_comments.sample(n=max_comments_per_topic, random_state=seed).copy()
    return topic_comments.sort_values(["score", "created_utc"], ascending=[False, False]).head(max_comments_per_topic).copy()


def normalize_excerpt(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def cue_score(text: str, lexicon: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for cue in lexicon if cue in lowered)


def comment_weight(score: int) -> float:
    return 1.0 + math.log1p(max(score, 0))


def cluster_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    if len(texts) < 3:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    matrix = vectorizer.fit_transform(texts)
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top_indices = mean_scores.argsort()[::-1]
    keywords: list[str] = []
    for index in top_indices:
        token = feature_names[index]
        token_parts = token.split()
        if len(token_parts) == 1 and token in STANCE_STOPWORDS:
            continue
        if any(part in STANCE_STOPWORDS for part in token_parts):
            continue
        score = float(mean_scores[index]) * (1.0 + 0.35 * min(len(token_parts), 3))
        if len(token_parts) == 1:
            score *= 0.7
        if score < 0.015:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) == top_n:
            break
    return keywords


def format_theme_list(themes: list[str], limit: int = 3) -> str:
    picked = [theme for theme in themes[:limit] if theme]
    if not picked:
        return "mixed lines of argument"
    if len(picked) == 1:
        return picked[0]
    if len(picked) == 2:
        return f"{picked[0]} and {picked[1]}"
    return f"{picked[0]}, {picked[1]}, and {picked[2]}"


def distinctive_themes(primary: list[str], secondary: list[str], limit: int = 3) -> list[str]:
    secondary_tokens = {item.lower() for item in secondary}
    distinct = [theme for theme in primary if theme.lower() not in secondary_tokens]
    return distinct[:limit] if distinct else primary[:limit]


def summarize_comment_frames(representative_comments: list[dict[str, Any]], limit: int = 2) -> str:
    excerpts = [normalize_excerpt(item["excerpt"], limit=120) for item in representative_comments[:limit] if item.get("excerpt")]
    if not excerpts:
        return ""
    return "; ".join(excerpts)


def generate_stance_summaries_generative(
    topic_label: str,
    support_keywords: list[str],
    opposing_keywords: list[str],
    support_representative_comments: list[dict[str, Any]],
    opposing_representative_comments: list[dict[str, Any]],
    disable_groq_summaries: bool,
) -> dict[str, str]:
    fallback = {
        "dominant_position_summary": infer_dominant_position(
            topic_label,
            support_keywords,
            support_representative_comments,
            opposing_keywords,
        ),
        "support_argument_summary": summarize_cluster_arguments(
            "support",
            topic_label,
            support_keywords,
            support_representative_comments,
            opposing_keywords,
        ),
        "opposing_argument_summary": summarize_cluster_arguments(
            "opposing",
            topic_label,
            opposing_keywords,
            opposing_representative_comments,
            support_keywords,
        ),
    }
    if disable_groq_summaries or not groq_available():
        return fallback

    try:
        system_prompt = (
            "You are summarizing stance clusters inside a political discussion topic. "
            "Return strict JSON with keys `dominant_position_summary`, `support_argument_summary`, and `opposing_argument_summary`. "
            "Each value should be 2 to 3 sentences, analytical, and grounded in the supplied themes and representative comments. "
            "Focus on the claims and reasoning patterns, not just repeated words."
        )
        support_examples = "\n".join(f"- {item.get('excerpt', '')}" for item in support_representative_comments[:4])
        opposing_examples = "\n".join(f"- {item.get('excerpt', '')}" for item in opposing_representative_comments[:4])
        user_prompt = (
            f"Topic label: {topic_label}\n"
            f"Support-side themes: {', '.join(support_keywords[:8])}\n"
            f"Opposing-side themes: {', '.join(opposing_keywords[:8])}\n"
            f"Support-side representative comments:\n{support_examples}\n\n"
            f"Opposing-side representative comments:\n{opposing_examples}\n\n"
            "Return JSON only."
        )
        result = groq_json_completion(system_prompt, user_prompt, max_tokens=520)
        cleaned = {
            key: normalize_excerpt(str(result.get(key, "")).replace("\n", " "), limit=520)
            for key in fallback
        }
        if all(cleaned.values()):
            return cleaned
        return fallback
    except Exception:
        return fallback


def representative_comment_records(
    comments: pd.DataFrame,
    embeddings: np.ndarray,
    centroid: np.ndarray,
    limit: int = MAX_REPRESENTATIVE_COMMENTS,
) -> list[dict[str, Any]]:
    centroid = centroid.reshape(1, -1)
    sims = cosine_similarity(embeddings, centroid).ravel()
    ranked = comments.copy()
    ranked["centroid_similarity"] = sims
    ranked["rank_score"] = ranked["score"].clip(lower=0) + ranked["centroid_similarity"] * 10.0
    ranked = ranked.sort_values(["rank_score", "score"], ascending=False).head(limit)
    return [
        {
            "comment_id": row.comment_id,
            "post_id": row.post_id,
            "author_hash": row.author_hash,
            "score": int(row.score),
            "stance_confidence": round(float(row.stance_confidence), 4),
            "excerpt": normalize_excerpt(row.body),
            "permalink": row.permalink,
        }
        for row in ranked.itertuples(index=False)
    ]


def summarize_cluster_arguments(
    stance_name: str,
    topic_label: str,
    keywords: list[str],
    representative_comments: list[dict[str, Any]],
    opposing_keywords: list[str],
) -> str:
    lead_themes = distinctive_themes(keywords, opposing_keywords, limit=3)
    contrast_themes = distinctive_themes(opposing_keywords, keywords, limit=2)
    evidence_text = summarize_comment_frames(representative_comments, limit=2)
    sentence = (
        f"Within the {topic_label.lower()} discussion, {stance_name}-side comments mainly argue through "
        f"{format_theme_list(lead_themes)}."
    )
    if contrast_themes:
        sentence += f" Compared with the other side, they put relatively more weight on {format_theme_list(contrast_themes, limit=2)}."
    if evidence_text:
        sentence += f" Representative comments repeatedly return to points such as {evidence_text}."
    return sentence


def infer_dominant_position(
    topic_label: str,
    dominant_keywords: list[str],
    representative_comments: list[dict[str, Any]],
    opposing_keywords: list[str],
) -> str:
    dominant_themes = distinctive_themes(dominant_keywords, opposing_keywords, limit=3)
    evidence_text = summarize_comment_frames(representative_comments, limit=2)
    sentence = (
        f"For {topic_label}, the dominant position is organized around {format_theme_list(dominant_themes)}."
    )
    if evidence_text:
        sentence += f" The strongest supporting comments keep circling back to points such as {evidence_text}."
    return sentence


def build_user_groups(topic_comments: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        topic_comments.groupby(["author_hash", "stance_label"])
        .agg(
            comments=("comment_id", "count"),
            total_score=("score", "sum"),
            avg_confidence=("stance_confidence", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return {"support_users": 0, "opposing_users": 0, "users": []}

    user_rows: list[dict[str, Any]] = []
    support_users = 0
    opposing_users = 0
    for author_hash, author_rows in grouped.groupby("author_hash"):
        author_rows = author_rows.sort_values(["comments", "total_score"], ascending=False)
        primary = author_rows.iloc[0]
        dominant_stance = primary["stance_label"]
        if dominant_stance == "support":
            support_users += 1
        else:
            opposing_users += 1
        user_rows.append(
            {
                "author_hash": author_hash,
                "dominant_stance": dominant_stance,
                "support_comments": int(author_rows.loc[author_rows["stance_label"] == "support", "comments"].sum()),
                "opposing_comments": int(author_rows.loc[author_rows["stance_label"] == "opposing", "comments"].sum()),
                "total_score": int(author_rows["total_score"].sum()),
                "avg_confidence": round(float(author_rows["avg_confidence"].mean()), 4),
            }
        )

    user_rows = sorted(
        user_rows,
        key=lambda item: (item["support_comments"] + item["opposing_comments"], item["total_score"]),
        reverse=True,
    )
    return {
        "support_users": support_users,
        "opposing_users": opposing_users,
        "users": user_rows[:MAX_USER_PREVIEW],
    }


def analyze_topic_comments(
    topic_row: dict[str, Any],
    topic_comments: pd.DataFrame,
    model,
    seed: int,
    disable_groq_summaries: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    texts = topic_comments["body"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    kmeans = MiniBatchKMeans(n_clusters=2, random_state=seed, batch_size=2048, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    topic_comments = topic_comments.copy()
    topic_comments["cluster_id"] = cluster_ids
    topic_comments["support_cue_score"] = topic_comments["body"].apply(lambda text: cue_score(text, SUPPORT_CUES))
    topic_comments["oppose_cue_score"] = topic_comments["body"].apply(lambda text: cue_score(text, OPPOSE_CUES))

    cluster_stats: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(topic_comments["cluster_id"].unique()):
        cluster_comments = topic_comments[topic_comments["cluster_id"] == cluster_id].copy()
        cluster_embeddings = embeddings[cluster_comments.index.to_numpy()]
        weights = cluster_comments["score"].apply(comment_weight).to_numpy()
        weighted_size = float(weights.sum())
        support_cues = float(cluster_comments["support_cue_score"].sum())
        oppose_cues = float(cluster_comments["oppose_cue_score"].sum())
        centroid = np.average(cluster_embeddings, axis=0, weights=weights)
        cluster_stats[int(cluster_id)] = {
            "count": int(len(cluster_comments)),
            "weighted_size": weighted_size,
            "support_cues": support_cues,
            "oppose_cues": oppose_cues,
            "centroid": centroid / np.linalg.norm(centroid),
        }

    dominant_cluster = max(
        cluster_stats,
        key=lambda cid: (
            cluster_stats[cid]["weighted_size"],
            cluster_stats[cid]["support_cues"] - cluster_stats[cid]["oppose_cues"],
        ),
    )
    opposing_cluster = next(cid for cid in cluster_stats if cid != dominant_cluster)

    centroids = np.vstack([cluster_stats[dominant_cluster]["centroid"], cluster_stats[opposing_cluster]["centroid"]])
    sim_matrix = cosine_similarity(embeddings, centroids)
    confidence = np.abs(sim_matrix[:, 0] - sim_matrix[:, 1])

    topic_comments["stance_label"] = np.where(topic_comments["cluster_id"] == dominant_cluster, "support", "opposing")
    topic_comments["stance_confidence"] = confidence

    support_comments = topic_comments[topic_comments["stance_label"] == "support"].copy()
    opposing_comments = topic_comments[topic_comments["stance_label"] == "opposing"].copy()

    support_embeddings = embeddings[support_comments.index.to_numpy()]
    opposing_embeddings = embeddings[opposing_comments.index.to_numpy()]
    support_keywords = cluster_keywords(support_comments["body"].tolist())
    opposing_keywords = cluster_keywords(opposing_comments["body"].tolist())

    support_reps = representative_comment_records(
        support_comments,
        support_embeddings,
        cluster_stats[dominant_cluster]["centroid"],
    )
    opposing_reps = representative_comment_records(
        opposing_comments,
        opposing_embeddings,
        cluster_stats[opposing_cluster]["centroid"],
    )

    user_groups = build_user_groups(topic_comments)
    disagreement_index = round(
        float(min(len(support_comments), len(opposing_comments)) / max(len(support_comments), len(opposing_comments))),
        4,
    )
    generated_summaries = generate_stance_summaries_generative(
        topic_label=topic_row["label"],
        support_keywords=support_keywords,
        opposing_keywords=opposing_keywords,
        support_representative_comments=support_reps,
        opposing_representative_comments=opposing_reps,
        disable_groq_summaries=disable_groq_summaries,
    )

    topic_summary = {
        "topic_id": int(topic_row["topic_id"]),
        "topic_label": topic_row["label"],
        "topic_keywords": topic_row["keywords"],
        "topic_share": float(topic_row["topic_share"]),
        "trend_type": topic_row["trend_type"],
        "post_count": int(topic_row["post_count"]),
        "comment_count": int(len(topic_comments)),
        "support_comment_count": int(len(support_comments)),
        "opposing_comment_count": int(len(opposing_comments)),
        "support_share": round(float(len(support_comments) / len(topic_comments)), 4),
        "opposing_share": round(float(len(opposing_comments) / len(topic_comments)), 4),
        "disagreement_index": disagreement_index,
        "dominant_position_summary": generated_summaries["dominant_position_summary"],
        "support_argument_summary": generated_summaries["support_argument_summary"],
        "opposing_argument_summary": generated_summaries["opposing_argument_summary"],
        "support_keywords": support_keywords,
        "opposing_keywords": opposing_keywords,
        "support_representative_comments": support_reps,
        "opposing_representative_comments": opposing_reps,
        "user_groups": user_groups,
        "top_flairs": topic_row["top_flairs"],
        "representative_posts": topic_row["representative_posts"],
    }

    return topic_comments, topic_summary


def save_outputs(
    out_dir: Path,
    all_comments: pd.DataFrame,
    topic_summaries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    comment_jsonl_path = out_dir / "comment_stances.jsonl"
    comment_csv_path = out_dir / "comment_stances.csv"
    summary_json_path = out_dir / "topic_stance_summary.json"
    summary_csv_path = out_dir / "topic_stance_summary.csv"
    user_groups_path = out_dir / "topic_user_stance_groups.json"
    metadata_path = out_dir / "run_metadata.json"

    comment_records = all_comments[
        [
            "comment_id",
            "post_id",
            "topic_id",
            "topic_label",
            "author_hash",
            "parent_id",
            "created_iso",
            "score",
            "stance_label",
            "stance_confidence",
            "cluster_id",
            "body",
            "permalink",
            "topic_post_title",
            "topic_post_permalink",
            "link_flair_text",
        ]
    ].copy()
    comment_records["stance_confidence"] = comment_records["stance_confidence"].round(4)
    comment_records.to_csv(comment_csv_path, index=False)
    with comment_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in comment_records.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_json_path.write_text(json.dumps(topic_summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    user_groups_rows = []
    for item in topic_summaries:
        summary_rows.append(
            {
                "topic_id": item["topic_id"],
                "topic_label": item["topic_label"],
                "comment_count": item["comment_count"],
                "support_comment_count": item["support_comment_count"],
                "opposing_comment_count": item["opposing_comment_count"],
                "support_share": item["support_share"],
                "opposing_share": item["opposing_share"],
                "disagreement_index": item["disagreement_index"],
                "dominant_position_summary": item["dominant_position_summary"],
                "support_argument_summary": item["support_argument_summary"],
                "opposing_argument_summary": item["opposing_argument_summary"],
                "support_keywords": ", ".join(item["support_keywords"]),
                "opposing_keywords": ", ".join(item["opposing_keywords"]),
            }
        )
        user_groups_rows.append(
            {
                "topic_id": item["topic_id"],
                "topic_label": item["topic_label"],
                "support_users": item["user_groups"]["support_users"],
                "opposing_users": item["user_groups"]["opposing_users"],
                "users": item["user_groups"]["users"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    user_groups_path.write_text(json.dumps(user_groups_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_cache_dir = Path(args.model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    configure_huggingface_env(model_cache_dir, args.disable_hf_ssl_verify)

    comments_path = resolve_input_path(args.comments, "data/cleaned/comments_clean.jsonl")
    post_topics_path = resolve_input_path(args.post_topics, "data/topic_analysis/post_topics.csv")
    topic_summary_path = resolve_input_path(args.topic_summary, "data/topic_analysis/topic_summary.json")

    topic_summary = load_topic_summary(topic_summary_path)
    valid_topic_ids = set(int(topic_id) for topic_id in topic_summary["topic_id"].tolist())
    post_topics = load_post_topics(post_topics_path, valid_topic_ids)
    comments = build_comment_frame(comments_path, post_topics)

    topic_summary_rows = {
        int(row["topic_id"]): row
        for row in topic_summary.to_dict(orient="records")
    }

    major_topic_ids = [
        topic_id
        for topic_id, count in comments.groupby("topic_id").size().to_dict().items()
        if count >= args.min_comments_per_topic
    ]
    requested_topic_ids = parse_topic_ids(args.topic_ids)
    if requested_topic_ids:
        major_topic_ids = [topic_id for topic_id in major_topic_ids if topic_id in requested_topic_ids]
    if args.max_topics and args.max_topics > 0:
        topic_counts = comments.groupby("topic_id").size().to_dict()
        major_topic_ids = sorted(
            major_topic_ids,
            key=lambda topic_id: topic_counts.get(topic_id, 0),
            reverse=True,
        )[: args.max_topics]
    major_topic_ids = sorted(major_topic_ids)
    if not major_topic_ids:
        raise ValueError("No topics met the minimum comment threshold.")

    model = load_sentence_model(args.embedding_model, model_cache_dir)

    topic_comment_frames: list[pd.DataFrame] = []
    topic_summaries: list[dict[str, Any]] = []

    for topic_id in major_topic_ids:
        topic_comments = comments[comments["topic_id"] == topic_id].copy()
        topic_comments = sample_topic_comments(
            topic_comments,
            max_comments_per_topic=args.max_comments_per_topic,
            sample_mode=args.sample_mode,
            seed=args.seed,
        ).reset_index(drop=True)
        topic_row = topic_summary_rows[int(topic_id)]
        topic_comments["topic_label"] = topic_row["label"]
        analyzed_comments, topic_info = analyze_topic_comments(
            topic_row,
            topic_comments,
            model,
            args.seed,
            args.disable_groq_summaries,
        )
        topic_comment_frames.append(analyzed_comments)
        topic_summaries.append(topic_info)

    all_comments = pd.concat(topic_comment_frames, ignore_index=True)
    topic_summaries = sorted(topic_summaries, key=lambda item: item["comment_count"], reverse=True)

    metadata = {
        "input_comments": str(comments_path),
        "input_post_topics": str(post_topics_path),
        "input_topic_summary": str(topic_summary_path),
        "output_dir": str(out_dir),
        "embedding_model": args.embedding_model,
        "model_cache_dir": str(model_cache_dir),
        "disable_hf_ssl_verify": args.disable_hf_ssl_verify,
        "disable_groq_summaries": args.disable_groq_summaries,
        "min_comments_per_topic": args.min_comments_per_topic,
        "topic_ids": sorted(requested_topic_ids),
        "max_topics": args.max_topics,
        "max_comments_per_topic": args.max_comments_per_topic,
        "sample_mode": args.sample_mode,
        "topic_count_analyzed": len(topic_summaries),
        "comment_count_analyzed": int(len(all_comments)),
        "outputs": {
            "comment_stances_jsonl": str(out_dir / "comment_stances.jsonl"),
            "comment_stances_csv": str(out_dir / "comment_stances.csv"),
            "topic_stance_summary_json": str(out_dir / "topic_stance_summary.json"),
            "topic_stance_summary_csv": str(out_dir / "topic_stance_summary.csv"),
            "topic_user_stance_groups_json": str(out_dir / "topic_user_stance_groups.json"),
            "metadata": str(out_dir / "run_metadata.json"),
        },
    }

    save_outputs(out_dir, all_comments, topic_summaries, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
