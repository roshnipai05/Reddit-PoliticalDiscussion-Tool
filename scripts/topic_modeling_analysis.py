"""Run topic modeling and trend analysis for cleaned Reddit posts.

This script targets Project Part 1 tasks 1.1, 1.2 and 1.3:
* 1.1: Show aggregate properties (users, posts, comments, score)
* 1.2: Identify 5-20 key topics with labels, keywords, and share of posts
* 1.3: Distinguish trending topics from persistent topics

BERTopic is used for clustering and c-TF-IDF topic keywords using 
SentenceTransformer semantic embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_SEED = 42
DEFAULT_TARGET_TOPICS = 15 # Hits the 5-20 target constraint
STOPWORDS = {
    "just", "like", "really", "want", "know", "time", "thing", "things",
    "going", "make", "feel", "think", "don", "does", "did", "ll", "ve",
    "got", "getting", "said", "say", "way", "day", "days", "year", "years",
    "old", "kid", "kids", "child", "children", "son", "daughter", "baby",
    "toddler", "parent", "parents", "parenting"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", default="data/cleaned/posts_clean.jsonl")
    parser.add_argument("--out-dir", default="data/topic_analysis")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-topics", type=int, default=DEFAULT_TARGET_TOPICS)
    parser.add_argument("--min-topic-size", type=int, default=50)
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for dense semantic embeddings.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default="data/models/huggingface",
        help="Local cache directory for transformer model downloads.",
    )
    parser.add_argument(
        "--disable-hf-ssl-verify",
        action="store_true",
        help="Disable SSL verification for Hugging Face downloads when local certificates are broken.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing topic outputs in out-dir and refresh only labels/report/chart.",
    )
    return parser.parse_args()


def load_posts(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            post = json.loads(line)
            text = str(post.get("text") or "").strip()
            if not text:
                continue
            records.append(
                {
                    "post_id": post.get("post_id"),
                    "created_utc": post.get("created_utc"),
                    "created_iso": post.get("created_iso"),
                    "created_month": pd.to_datetime(post.get("created_iso"), utc=True).strftime("%Y-%m"),
                    "author_hash": post.get("author_hash", "Unknown"),
                    "title": post.get("title"),
                    "text": text,
                    "link_flair_text": post.get("link_flair_text") or "Unspecified",
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "permalink": post.get("permalink"),
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No posts found in {path}")
    return df


def generate_aggregate_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Spec 1.1: Calculate aggregate properties of the database."""
    return {
        "total_posts": len(df),
        "total_unique_users": df["author_hash"].nunique(),
        "total_comments": int(df["num_comments"].sum()),
        "total_upvotes": int(df["score"].sum()),
        "date_range_start": df["created_month"].min(),
        "date_range_end": df["created_month"].max()
    }


def fit_bertopic(
    texts: list[str],
    embedding_model: str,
    model_cache_dir: str,
    seed: int,
    target_topics: int,
    min_topic_size: int,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    
    sentence_model = SentenceTransformer(embedding_model, cache_folder=model_cache_dir)

    model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=target_topics, # Spec 1.2: Limit to 5-20 topics
        top_n_words=10,          # Spec 1.2: Limit keywords
        calculate_probabilities=False,
        verbose=True,
    )
    
    topics, probabilities = model.fit_transform(texts)
    return model, topics, probabilities


def configure_huggingface_client(disable_ssl_verify: bool) -> None:
    if not disable_ssl_verify:
        return

    from huggingface_hub import set_client_factory

    def client_factory() -> httpx.Client:
        return httpx.Client(verify=False, follow_redirects=True, timeout=60.0)

    set_client_factory(client_factory)


def keyword_list(topic_model: Any, topic_id: int, limit: int = 10) -> list[str]:
    """Spec 1.2: Get top 5-10 keywords per topic."""
    words = topic_model.get_topic(topic_id) or []
    keywords: list[str] = []
    for word, _ in words:
        clean = word.strip()
        if clean and clean not in keywords:
            keywords.append(clean)
        if len(keywords) == limit:
            break
    return keywords


def make_label(keywords: list[str]) -> str:
    """Spec 1.2: Create a short descriptive label based on top keywords."""
    if not keywords:
        return "Other Discussion"
    selected: list[str] = []
    for keyword in keywords:
        for part in re.split(r"\s+", keyword):
            if part not in STOPWORDS and part not in selected:
                selected.append(part)
            if len(selected) == 3:
                break
        if len(selected) == 3:
            break
    if not selected:
        selected = keywords[:2]
    return " / ".join(word.title() for word in selected)


def classify_topic_trend(monthly_counts: pd.Series, total_posts_by_month: pd.Series) -> dict[str, Any]:
    """Spec 1.3: Distinguish between trending and persistent topics mathematically."""
    aligned_counts = monthly_counts.reindex(total_posts_by_month.index, fill_value=0)
    monthly_share = aligned_counts / total_posts_by_month
    active_months = int((aligned_counts > 0).sum())
    total_months = int(len(total_posts_by_month))
    
    recent_window = max(1, min(2, total_months // 3))
    early_window = max(1, min(2, total_months // 3))
    recent_share = float(monthly_share.tail(recent_window).mean())
    early_share = float(monthly_share.head(early_window).mean())
    overall_share = float(aligned_counts.sum() / total_posts_by_month.sum())
    
    x = np.arange(total_months)
    slope = float(np.polyfit(x, monthly_share.to_numpy(), 1)[0]) if total_months > 1 else 0.0
    cv = float(monthly_share.std() / monthly_share.mean()) if monthly_share.mean() > 0 else 0.0

    stable_gap = abs(recent_share - overall_share) / max(overall_share, 1e-9)
    slope_threshold = 0.00035

    if recent_share >= max(overall_share * 1.15, early_share * 1.3) and slope > slope_threshold:
        label = "Trending"
    elif early_share >= max(overall_share * 1.15, recent_share * 1.3) and slope < -slope_threshold:
        label = "Declining"
    elif active_months >= max(5, int(0.75 * total_months)) and cv <= 0.35 and stable_gap <= 0.2:
        label = "Persistent"
    else:
        label = "Episodic"

    return {
        "trend_type": label,
        "active_months": active_months,
        "total_months": total_months,
        "recent_share": recent_share,
        "early_share": early_share,
        "trend_slope": slope,
        "share_cv": cv,
    }


def classify_topic_trend_from_metrics(
    active_months: int,
    total_months: int,
    recent_share: float,
    early_share: float,
    overall_share: float,
    slope: float,
    cv: float,
) -> str:
    stable_gap = abs(recent_share - overall_share) / max(overall_share, 1e-9)
    slope_threshold = 0.00035

    if recent_share >= max(overall_share * 1.15, early_share * 1.3) and slope > slope_threshold:
        return "Trending"
    if early_share >= max(overall_share * 1.15, recent_share * 1.3) and slope < -slope_threshold:
        return "Declining"
    if active_months >= max(5, int(0.75 * total_months)) and cv <= 0.35 and stable_gap <= 0.2:
        return "Persistent"
    return "Episodic"


def flair_breakdown(topic_posts: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    counts = topic_posts["link_flair_text"].fillna("Unspecified").value_counts().head(top_n)
    total = len(topic_posts)
    return [
        {"flair": flair, "posts": int(count), "share_within_topic": round(float(count / total), 4)}
        for flair, count in counts.items()
    ]


def representative_posts(topic_posts: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
    cols = ["post_id", "title", "score", "num_comments", "created_month", "permalink"]
    sample = topic_posts.sort_values(["score", "num_comments"], ascending=False).head(top_n)
    return sample[cols].to_dict(orient="records")


def write_report(summary: pd.DataFrame, agg_stats: dict, out_path: Path) -> None:
    """Generate a Markdown report outputting Spec 1.1 and 1.2 requirements."""
    lines = [
        "# Reddit Subreddit Analysis Report",
        "",
        "## Part 1.1: Aggregate Database Properties",
        f"- **Total Posts:** {agg_stats['total_posts']:,}",
        f"- **Total Unique Users:** {agg_stats['total_unique_users']:,}",
        f"- **Total Comments:** {agg_stats['total_comments']:,}",
        f"- **Total Upvotes:** {agg_stats['total_upvotes']:,}",
        f"- **Date Range:** {agg_stats['date_range_start']} to {agg_stats['date_range_end']}",
        "",
        "## Part 1.2 & 1.3: Topic Analysis (Top Conversation Themes)",
        "",
    ]
    
    for row in summary.sort_values("topic_share", ascending=False).to_dict(orient="records"):
        keywords = ", ".join(row["keywords"])
        flair_text = ", ".join(
            f"{item['flair']} ({item['share_within_topic']:.1%})" for item in row["top_flairs"]
        ) or "Unspecified"
        lines.extend(
            [
                f"### {row['label']} [{row['trend_type']}]",
                "",
                f"- **Topic ID:** `{row['topic_id']}`",
                f"- **Share of Total Posts:** {row['topic_share']:.2%} ({row['post_count']} posts)",
                f"- **Top Keywords:** {keywords}",
                f"- **Top Flair Filters:** {flair_text}",
                f"- **Active Months:** {row['active_months']} of {row['total_months']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary["keywords"] = summary["keywords"].apply(
        lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
    )
    for column in ("top_flairs", "representative_posts"):
        summary[column] = summary[column].apply(json.loads)
    return summary


def refresh_trend_labels(summary: pd.DataFrame) -> pd.DataFrame:
    refreshed = summary.copy()
    refreshed["trend_type"] = refreshed.apply(
        lambda row: classify_topic_trend_from_metrics(
            active_months=int(row["active_months"]),
            total_months=int(row["total_months"]),
            recent_share=float(row["recent_share"]),
            early_share=float(row["early_share"]),
            overall_share=float(row["topic_share"]),
            slope=float(row["trend_slope"]),
            cv=float(row["share_cv"]),
        ),
        axis=1,
    )
    return refreshed


def save_summary_outputs(summary: pd.DataFrame, out_dir: Path, agg_stats: dict) -> None:
    topic_summary_csv_path = out_dir / "topic_summary.csv"
    topic_summary_json_path = out_dir / "topic_summary.json"
    report_path = out_dir / "topic_analysis_report.md"
    html_path = out_dir / "topic_share_chart.html"

    summary_for_csv = summary.copy()
    summary_for_csv["keywords"] = summary_for_csv["keywords"].apply(lambda values: ", ".join(values))
    summary_for_csv["top_flairs"] = summary_for_csv["top_flairs"].apply(json.dumps)
    summary_for_csv["representative_posts"] = summary_for_csv["representative_posts"].apply(json.dumps)
    summary_for_csv.to_csv(topic_summary_csv_path, index=False)
    topic_summary_json_path.write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(summary, agg_stats, report_path)

    chart_df = summary.sort_values("topic_share", ascending=True).copy()
    chart_df["keywords_text"] = chart_df["keywords"].apply(lambda values: ", ".join(values[:5]))
    fig = px.bar(
        chart_df,
        x="topic_share",
        y="label",
        color="trend_type",
        orientation="h",
        hover_data=["post_count", "keywords_text"],
        labels={"topic_share": "Share of all posts", "label": "Topic"},
        title="Key Topics in Subreddit Posts",
    )
    fig.update_layout(xaxis_tickformat=".0%", height=max(500, 40 * len(chart_df) + 220))
    fig.write_html(html_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = Path(args.model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(model_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache_dir))
    configure_huggingface_client(args.disable_hf_ssl_verify)

    # Load Data
    posts = load_posts(Path(args.posts))
    texts = posts["text"].tolist()
    
    # Generate Spec 1.1 Aggregates
    agg_stats = generate_aggregate_stats(posts)
    (out_dir / "aggregate_stats.json").write_text(json.dumps(agg_stats, indent=2))
    metadata_path = out_dir / "run_metadata.json"

    if args.reuse_existing:
        summary = load_existing_summary(out_dir / "topic_summary.csv")
        summary = refresh_trend_labels(summary).sort_values("topic_share", ascending=False)
        save_summary_outputs(summary, out_dir, agg_stats)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["reuse_existing"] = True
        metadata["trend_labels_refreshed"] = True
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Refreshed trend labels for {len(summary)} topics using existing outputs.")
        print(f"Report generated at: {out_dir / 'topic_analysis_report.md'}")
        print(f"Metadata generated at: {metadata_path}")
        return

    # Run BERTopic (Spec 1.2)
    topic_model, topics, _ = fit_bertopic(
        texts=texts,
        embedding_model=args.embedding_model,
        model_cache_dir=str(model_cache_dir),
        seed=args.seed,
        target_topics=args.target_topics,
        min_topic_size=args.min_topic_size,
    )
    posts["topic_id"] = topics
    non_outlier = posts[posts["topic_id"] != -1].copy()
    if non_outlier.empty:
        raise RuntimeError("BERTopic produced only outlier assignments. Lower --min-topic-size and rerun.")

    total_posts = len(posts)
    months = pd.Index(sorted(posts["created_month"].unique()), name="created_month")
    total_posts_by_month = posts.groupby("created_month").size().reindex(months, fill_value=0)

    # Compile Topic Metadata
    summary_rows: list[dict[str, Any]] = []
    for topic_id in sorted(non_outlier["topic_id"].unique()):
        topic_posts = posts[posts["topic_id"] == topic_id].copy()
        keywords = keyword_list(topic_model, int(topic_id), limit=10)
        trend = classify_topic_trend(topic_posts.groupby("created_month").size(), total_posts_by_month)
        
        summary_rows.append(
            {
                "topic_id": int(topic_id),
                "label": make_label(keywords),
                "keywords": keywords,
                "post_count": int(len(topic_posts)),
                "topic_share": float(len(topic_posts) / total_posts), # Spec 1.2: Share of posts
                "top_flairs": flair_breakdown(topic_posts),
                "representative_posts": representative_posts(topic_posts),
                **trend,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("topic_share", ascending=False)

    # Save Outputs
    post_topics_path = out_dir / "post_topics.csv"
    topic_summary_csv_path = out_dir / "topic_summary.csv"
    topic_summary_json_path = out_dir / "topic_summary.json"
    monthly_path = out_dir / "topic_monthly_trends.csv"
    flair_path = out_dir / "topic_flair_breakdown.csv"
    report_path = out_dir / "topic_analysis_report.md"
    html_path = out_dir / "topic_share_chart.html"

    posts[
        [
            "post_id",
            "topic_id",
            "created_month",
            "link_flair_text",
            "score",
            "num_comments",
            "title",
            "permalink",
        ]
    ].to_csv(post_topics_path, index=False)

    save_summary_outputs(summary, out_dir, agg_stats)

    monthly_rows = []
    for topic_id in sorted(non_outlier["topic_id"].unique()):
        topic_posts = posts[posts["topic_id"] == topic_id]
        counts = topic_posts.groupby("created_month").size().reindex(months, fill_value=0)
        for month, count in counts.items():
            monthly_rows.append(
                {
                    "topic_id": int(topic_id),
                    "created_month": month,
                    "post_count": int(count),
                    "topic_month_share": float(count / total_posts_by_month.loc[month]),
                }
            )
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)

    flair_rows = []
    for row in summary.to_dict(orient="records"):
        for item in row["top_flairs"]:
            flair_rows.append({"topic_id": row["topic_id"], "label": row["label"], **item})
    pd.DataFrame(flair_rows).to_csv(flair_path, index=False)

    metadata = {
        "input_posts": args.posts,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "target_topics": args.target_topics,
        "min_topic_size": args.min_topic_size,
        "embedding_model": args.embedding_model,
        "model_cache_dir": str(model_cache_dir),
        "disable_hf_ssl_verify": args.disable_hf_ssl_verify,
        "post_count": total_posts,
        "assigned_non_outlier_posts": int(len(non_outlier)),
        "outlier_posts": int((posts["topic_id"] == -1).sum()),
        "topic_count": int(len(summary)),
        "outputs": {
            "aggregate_stats": str(out_dir / "aggregate_stats.json"),
            "post_topics": str(post_topics_path),
            "topic_summary_csv": str(topic_summary_csv_path),
            "topic_summary_json": str(topic_summary_json_path),
            "monthly_trends": str(monthly_path),
            "flair_breakdown": str(flair_path),
            "report": str(report_path),
            "chart": str(html_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Analysis complete. Found {len(summary)} core topics.")
    print(f"Report generated at: {report_path}")
    print(f"Metadata generated at: {metadata_path}")

if __name__ == "__main__":
    main()
