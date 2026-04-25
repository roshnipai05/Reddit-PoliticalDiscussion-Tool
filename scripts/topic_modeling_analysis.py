"""Run topic modeling and trend analysis for cleaned Reddit posts.

This script covers Part 1.1, 1.2, and 1.3 of the project spec:
- aggregate subreddit properties
- 5-20 interpretable topics with labels, descriptions, keywords, and post share
- trending vs persistent topic diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
DEFAULT_TARGET_TOPICS = 15
DEFAULT_MODEL = "all-MiniLM-L6-v2"
TOPIC_WORD_LIMIT = 10
TITLE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
POLITICAL_STOPWORDS = {
    "amp",
    "answer",
    "anyone",
    "anybody",
    "argue",
    "argument",
    "arguments",
    "candidate",
    "candidates",
    "country",
    "countries",
    "democrat",
    "democrats",
    "discussion",
    "election",
    "elections",
    "government",
    "issue",
    "issues",
    "people",
    "political",
    "politics",
    "question",
    "republican",
    "republicans",
    "state",
    "states",
    "think",
    "trump",
    "biden",
    "harris",
    "voters",
    "vote",
    "voting",
    "want",
    "would",
}
GENERIC_TOPIC_STOPWORDS = ENGLISH_STOP_WORDS.union(POLITICAL_STOPWORDS)
UPPER_TOKENS = {"US", "UK", "EU", "UN", "NATO", "NLRB", "DNC", "RNC", "GOP", "IRS", "DEA"}
MAJOR_TOPIC_RULES = [
    (
        "elections-campaigns",
        "Elections & Campaigns",
        {"us elections", "us politics"},
        {
            "ballot",
            "campaign",
            "debate",
            "delegate",
            "electoral",
            "election",
            "electability",
            "poll",
            "primary",
            "ticket",
            "turnout",
            "vice president",
        },
    ),
    (
        "institutions-law",
        "Institutions & Law",
        {"legal/courts", "legislation"},
        {
            "agency",
            "constitution",
            "congress",
            "court",
            "courts",
            "executive",
            "law",
            "legal",
            "ruling",
            "senate",
            "supreme court",
            "chevron",
        },
    ),
    (
        "foreign-policy",
        "Foreign Affairs",
        {"international politics", "european politics", "non-us politics"},
        {
            "china",
            "gaza",
            "hamas",
            "iran",
            "israel",
            "nato",
            "palestine",
            "russia",
            "ukraine",
            "war",
            "world",
        },
    ),
    (
        "economy-policy",
        "Economy & Domestic Policy",
        {"us politics", "legislation"},
        {
            "abortion",
            "economy",
            "healthcare",
            "housing",
            "immigration",
            "inflation",
            "jobs",
            "labor",
            "prices",
            "tax",
            "wages",
        },
    ),
]


def resolve_input_path(provided: str | None, default_name: str) -> Path:
    if provided:
        return Path(provided)
    path = ROOT / "data" / "cleaned" / default_name
    if path.exists():
        return path
    raise FileNotFoundError(f"Input not found at {path}. Pass the path explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", default=None)
    parser.add_argument("--out-dir", default="data/topic_analysis")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-topics", type=int, default=DEFAULT_TARGET_TOPICS)
    parser.add_argument("--min-topic-size", type=int, default=90)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--model-cache-dir", default="data/models/huggingface")
    parser.add_argument("--disable-hf-ssl-verify", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--reduce-outliers",
        action="store_true",
        help="Reassign BERTopic outliers to the nearest topic embedding.",
    )
    return parser.parse_args()


def load_posts(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            post = json.loads(line)
            text = str(post.get("text") or "").strip()
            title = str(post.get("title") or "").strip()
            if not text or not title:
                continue
            created_iso = post.get("created_iso")
            created_dt = pd.to_datetime(created_iso, utc=True)
            rows.append(
                {
                    "post_id": post.get("post_id"),
                    "created_utc": post.get("created_utc"),
                    "created_iso": created_iso,
                    "created_month": created_dt.strftime("%Y-%m"),
                    "created_date": created_dt.strftime("%Y-%m-%d"),
                    "author_hash": post.get("author_hash", "Unknown"),
                    "title": title,
                    "selftext": post.get("selftext", ""),
                    "text": text,
                    "text_length": len(text),
                    "link_flair_text": post.get("link_flair_text") or "Unspecified",
                    "score": int(post.get("score") or 0),
                    "num_comments": int(post.get("num_comments") or 0),
                    "permalink": post.get("permalink"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No posts found in {path}")
    return df


def generate_aggregate_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_posts": int(len(df)),
        "total_unique_users": int(df["author_hash"].nunique()),
        "total_comments": int(df["num_comments"].sum()),
        "total_upvotes": int(df["score"].sum()),
        "date_range_start": str(df["created_month"].min()),
        "date_range_end": str(df["created_month"].max()),
        "top_flairs": [
            {"flair": flair, "posts": int(count)}
            for flair, count in df["link_flair_text"].value_counts().head(8).items()
        ],
    }


def configure_huggingface_client(disable_ssl_verify: bool) -> None:
    if not disable_ssl_verify:
        return
    from huggingface_hub import set_client_factory

    def client_factory() -> httpx.Client:
        return httpx.Client(verify=False, follow_redirects=True, timeout=60.0)

    set_client_factory(client_factory)


def fit_topic_model(
    texts: list[str],
    embedding_model: str,
    model_cache_dir: str,
    seed: int,
    target_topics: int,
    min_topic_size: int,
    reduce_outliers: bool,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    stop_words = sorted(GENERIC_TOPIC_STOPWORDS)
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 3),
        min_df=4,
        max_df=0.85,
    )
    umap_model = UMAP(
        n_neighbors=18,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=max(8, min_topic_size // 8),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    sentence_model = SentenceTransformer(embedding_model, cache_folder=model_cache_dir)
    embeddings = sentence_model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=target_topics,
        top_n_words=TOPIC_WORD_LIMIT,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probabilities = model.fit_transform(texts, embeddings)
    if reduce_outliers and any(topic == -1 for topic in topics):
        reduced_topics = model.reduce_outliers(
            texts,
            topics,
            strategy="embeddings",
            embeddings=embeddings,
        )
        model.update_topics(texts, topics=reduced_topics, vectorizer_model=vectorizer)
        topics = reduced_topics
    return model, embeddings, topics, probabilities


def keyword_list(topic_model: Any, topic_id: int, limit: int = TOPIC_WORD_LIMIT) -> list[str]:
    words = topic_model.get_topic(topic_id) or []
    result: list[str] = []
    for word, _ in words:
        word = word.strip()
        if word and word not in result and word not in GENERIC_TOPIC_STOPWORDS:
            result.append(word)
        if len(result) >= limit:
            break
    return result


def title_tokens(text: str) -> list[str]:
    return [token.lower() for token in TITLE_TOKEN_RE.findall(text)]


def score_title_phrases(topic_posts: pd.DataFrame, top_n: int = 12) -> list[str]:
    phrases: Counter[str] = Counter()
    for title in topic_posts["title"].head(200):
        tokens = [token for token in title_tokens(title) if token not in GENERIC_TOPIC_STOPWORDS]
        for n in (3, 2):
            for idx in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[idx : idx + n])
                if any(part in GENERIC_TOPIC_STOPWORDS for part in phrase.split()):
                    continue
                phrases[phrase] += 1
    ranked = [
        phrase
        for phrase, _ in phrases.most_common()
        if not phrase.isdigit() and len(phrase) >= 6
    ]
    return ranked[:top_n]


def format_label_text(text: str) -> str:
    parts = []
    for token in text.split():
        upper = token.upper()
        parts.append(upper if upper in UPPER_TOKENS else token.capitalize())
    return " ".join(parts)


def pick_label_phrase(topic_posts: pd.DataFrame, keywords: list[str]) -> str:
    phrases = score_title_phrases(topic_posts)
    if phrases:
        return format_label_text(phrases[0])

    picked: list[str] = []
    for keyword in keywords:
        for token in keyword.split():
            token = token.strip().lower()
            if token and token not in GENERIC_TOPIC_STOPWORDS and token not in picked:
                picked.append(token)
            if len(picked) == 3:
                break
        if len(picked) == 3:
            break
    if picked:
        return format_label_text(" ".join(picked))
    return "General Political Discussion"


def summarize_titles(topic_posts: pd.DataFrame, limit: int = 3) -> list[str]:
    rows = topic_posts.sort_values(["score", "num_comments"], ascending=False).head(limit)
    return [str(title) for title in rows["title"].tolist()]


def flair_breakdown(topic_posts: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    counts = topic_posts["link_flair_text"].fillna("Unspecified").value_counts().head(top_n)
    total = max(len(topic_posts), 1)
    return [
        {
            "flair": flair,
            "posts": int(count),
            "share_within_topic": round(float(count / total), 4),
        }
        for flair, count in counts.items()
    ]


def representative_posts(topic_posts: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    cols = [
        "post_id",
        "title",
        "score",
        "num_comments",
        "created_month",
        "link_flair_text",
        "permalink",
    ]
    sample = topic_posts.sort_values(
        ["score", "num_comments", "text_length"],
        ascending=False,
    ).head(top_n)
    return sample[cols].to_dict(orient="records")


def classify_topic_trend(monthly_counts: pd.Series, total_posts_by_month: pd.Series) -> dict[str, Any]:
    aligned_counts = monthly_counts.reindex(total_posts_by_month.index, fill_value=0)
    monthly_share = aligned_counts / total_posts_by_month
    active_months = int((aligned_counts > 0).sum())
    total_months = int(len(total_posts_by_month))
    recent_window = max(1, math.ceil(total_months / 3))
    early_window = recent_window
    recent_share = float(monthly_share.tail(recent_window).mean())
    early_share = float(monthly_share.head(early_window).mean())
    overall_share = float(aligned_counts.sum() / max(total_posts_by_month.sum(), 1))
    x = np.arange(total_months)
    slope = float(np.polyfit(x, monthly_share.to_numpy(), 1)[0]) if total_months > 1 else 0.0
    mean_share = float(monthly_share.mean())
    cv = float(monthly_share.std() / mean_share) if mean_share > 0 else 0.0
    lift = recent_share / max(early_share, 1e-9)

    if active_months >= max(5, int(0.75 * total_months)) and cv <= 0.4 and abs(recent_share - overall_share) <= overall_share * 0.2:
        label = "Persistent"
    elif recent_share >= max(overall_share * 1.2, early_share * 1.35) and slope > 0:
        label = "Trending"
    elif early_share >= max(overall_share * 1.2, recent_share * 1.35) and slope < 0:
        label = "Declining"
    else:
        label = "Episodic"

    return {
        "trend_type": label,
        "active_months": active_months,
        "total_months": total_months,
        "recent_share": round(recent_share, 6),
        "early_share": round(early_share, 6),
        "trend_slope": round(slope, 6),
        "share_cv": round(cv, 6),
        "trend_lift": round(lift, 4),
        "monthly_post_count_series": [int(value) for value in aligned_counts.tolist()],
        "monthly_share_series": [round(float(value), 6) for value in monthly_share.tolist()],
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
    if active_months >= max(5, int(0.75 * total_months)) and cv <= 0.4 and abs(recent_share - overall_share) <= overall_share * 0.2:
        return "Persistent"
    if recent_share >= max(overall_share * 1.2, early_share * 1.35) and slope > 0:
        return "Trending"
    if early_share >= max(overall_share * 1.2, recent_share * 1.35) and slope < 0:
        return "Declining"
    return "Episodic"


def detect_major_topic(topic_label: str, topic_keywords: list[str], top_flairs: list[dict[str, Any]]) -> tuple[str, str]:
    text = " ".join([topic_label.lower(), *[keyword.lower() for keyword in topic_keywords]])
    flair_names = {item["flair"].lower() for item in top_flairs}
    best_slug = "parties-ideology"
    best_name = "Parties & Public Narratives"
    best_score = 0
    for slug, name, flair_hints, keyword_hints in MAJOR_TOPIC_RULES:
        score = 0
        score += len(flair_names.intersection(flair_hints)) * 3
        score += sum(1 for hint in keyword_hints if hint in text)
        if score > best_score:
            best_score = score
            best_slug = slug
            best_name = name
    return best_slug, best_name


def build_topic_description(
    label: str,
    major_topic: str,
    keywords: list[str],
    top_flairs: list[dict[str, Any]],
    representative_titles: list[str],
) -> str:
    focus_terms = ", ".join(keywords[:4]) if keywords else "closely related issues"
    flair_text = top_flairs[0]["flair"] if top_flairs else "mixed flairs"
    title_hint = representative_titles[0] if representative_titles else label
    return (
        f"This topic groups posts about {label.lower()} within the broader {major_topic.lower()} discussion space. "
        f"It is primarily surfaced through {flair_text.lower()} posts and focuses on {focus_terms}. "
        f"Representative threads center on questions like: {title_hint}"
    )


def write_report(summary: pd.DataFrame, agg_stats: dict[str, Any], out_path: Path) -> None:
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
        "## Part 1.2 & 1.3: Topics",
        "",
    ]
    for row in summary.sort_values("topic_share", ascending=False).to_dict(orient="records"):
        flair_text = ", ".join(
            f"{item['flair']} ({item['share_within_topic']:.1%})" for item in row["top_flairs"]
        ) or "Unspecified"
        lines.extend(
            [
                f"### {row['label']} [{row['trend_type']}]",
                "",
                f"- **Major Topic:** {row['major_topic']}",
                f"- **Topic ID:** `{row['topic_id']}`",
                f"- **Share of Total Posts:** {row['topic_share']:.2%} ({row['post_count']} posts)",
                f"- **Description:** {row['topic_description']}",
                f"- **Top Keywords:** {', '.join(row['keywords'])}",
                f"- **Top Flair Filters:** {flair_text}",
                f"- **Trend Lift:** {row['trend_lift']:.2f}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary["keywords"] = summary["keywords"].apply(
        lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
    )
    json_columns = {
        "top_flairs",
        "representative_posts",
        "representative_titles",
        "monthly_post_count_series",
        "monthly_share_series",
    }
    for column in json_columns.intersection(summary.columns):
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


def save_summary_outputs(summary: pd.DataFrame, out_dir: Path, agg_stats: dict[str, Any]) -> None:
    topic_summary_csv_path = out_dir / "topic_summary.csv"
    topic_summary_json_path = out_dir / "topic_summary.json"
    report_path = out_dir / "topic_analysis_report.md"
    html_path = out_dir / "topic_share_chart.html"

    summary_for_csv = summary.copy()
    for column in ("keywords",):
        summary_for_csv[column] = summary_for_csv[column].apply(lambda values: ", ".join(values))
    for column in (
        "top_flairs",
        "representative_posts",
        "representative_titles",
        "monthly_post_count_series",
        "monthly_share_series",
    ):
        summary_for_csv[column] = summary_for_csv[column].apply(json.dumps)
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
        hover_data=["major_topic", "post_count", "keywords_text"],
        labels={"topic_share": "Share of all posts", "label": "Topic"},
        title="Key Topics in Subreddit Posts",
    )
    fig.update_layout(xaxis_tickformat=".0%", height=max(500, 42 * len(chart_df) + 220))
    fig.write_html(html_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    posts_path = resolve_input_path(args.posts, "posts_clean.jsonl")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = Path(args.model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(model_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache_dir))
    configure_huggingface_client(args.disable_hf_ssl_verify)

    posts = load_posts(posts_path)
    texts = posts["text"].tolist()
    agg_stats = generate_aggregate_stats(posts)
    (out_dir / "aggregate_stats.json").write_text(
        json.dumps(agg_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
        return

    topic_model, embeddings, topics, _ = fit_topic_model(
        texts=texts,
        embedding_model=args.embedding_model,
        model_cache_dir=str(model_cache_dir),
        seed=args.seed,
        target_topics=args.target_topics,
        min_topic_size=args.min_topic_size,
        reduce_outliers=args.reduce_outliers,
    )
    posts["topic_id"] = topics
    non_outlier = posts[posts["topic_id"] != -1].copy()
    if non_outlier.empty:
        raise RuntimeError("BERTopic produced only outlier assignments. Lower --min-topic-size and rerun.")

    total_posts = len(posts)
    months = pd.Index(sorted(posts["created_month"].unique()), name="created_month")
    total_posts_by_month = posts.groupby("created_month").size().reindex(months, fill_value=0)

    summary_rows: list[dict[str, Any]] = []
    for topic_id in sorted(non_outlier["topic_id"].unique()):
        topic_posts = posts[posts["topic_id"] == topic_id].copy()
        top_flairs = flair_breakdown(topic_posts)
        keywords = keyword_list(topic_model, int(topic_id), limit=TOPIC_WORD_LIMIT)
        label = pick_label_phrase(topic_posts, keywords)
        representative_title_list = summarize_titles(topic_posts, limit=3)
        major_topic_slug, major_topic = detect_major_topic(label, keywords, top_flairs)
        trend = classify_topic_trend(topic_posts.groupby("created_month").size(), total_posts_by_month)
        summary_rows.append(
            {
                "topic_id": int(topic_id),
                "label": label,
                "topic_description": build_topic_description(
                    label=label,
                    major_topic=major_topic,
                    keywords=keywords,
                    top_flairs=top_flairs,
                    representative_titles=representative_title_list,
                ),
                "major_topic_slug": major_topic_slug,
                "major_topic": major_topic,
                "keywords": keywords,
                "post_count": int(len(topic_posts)),
                "topic_share": float(len(topic_posts) / total_posts),
                "top_flairs": top_flairs,
                "representative_titles": representative_title_list,
                "representative_posts": representative_posts(topic_posts),
                **trend,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("topic_share", ascending=False)

    post_topics_path = out_dir / "post_topics.csv"
    monthly_path = out_dir / "topic_monthly_trends.csv"
    flair_path = out_dir / "topic_flair_breakdown.csv"

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

    monthly_rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        for month, count, share in zip(months.tolist(), row["monthly_post_count_series"], row["monthly_share_series"]):
            monthly_rows.append(
                {
                    "topic_id": row["topic_id"],
                    "label": row["label"],
                    "created_month": month,
                    "post_count": int(count),
                    "topic_month_share": float(share),
                }
            )
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)

    flair_rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        for item in row["top_flairs"]:
            flair_rows.append({"topic_id": row["topic_id"], "label": row["label"], **item})
    pd.DataFrame(flair_rows).to_csv(flair_path, index=False)

    metadata = {
        "input_posts": str(posts_path),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "target_topics": args.target_topics,
        "min_topic_size": args.min_topic_size,
        "embedding_model": args.embedding_model,
        "model_cache_dir": str(model_cache_dir),
        "disable_hf_ssl_verify": args.disable_hf_ssl_verify,
        "reduce_outliers": args.reduce_outliers,
        "post_count": int(total_posts),
        "assigned_non_outlier_posts": int(len(non_outlier)),
        "outlier_posts": int((posts["topic_id"] == -1).sum()),
        "topic_count": int(len(summary)),
        "month_axis": months.tolist(),
        "outputs": {
            "aggregate_stats": str(out_dir / "aggregate_stats.json"),
            "post_topics": str(post_topics_path),
            "topic_summary_csv": str(out_dir / "topic_summary.csv"),
            "topic_summary_json": str(out_dir / "topic_summary.json"),
            "monthly_trends": str(monthly_path),
            "flair_breakdown": str(flair_path),
            "report": str(out_dir / "topic_analysis_report.md"),
            "chart": str(out_dir / "topic_share_chart.html"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analysis complete. Found {len(summary)} core topics.")


if __name__ == "__main__":
    main()
