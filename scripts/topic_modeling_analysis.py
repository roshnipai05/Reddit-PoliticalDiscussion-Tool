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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from llm_summaries import groq_available, groq_json_completion


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
DOMAIN_ARCHETYPES = [
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
        "Institutions, Courts & Law",
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
        "economy-domestic-policy",
        "Economy, Labor & Domestic Policy",
        {"us politics", "legislation"},
        {
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
            "worker",
        },
    ),
    (
        "identity-rights-social-conflict",
        "Identity, Rights & Social Conflict",
        {"us politics", "political theory"},
        {
            "abortion",
            "civil rights",
            "gender",
            "identity",
            "police",
            "protest",
            "race",
            "racial",
            "speech",
            "trans",
            "women",
        },
    ),
    (
        "foreign-policy-geopolitics",
        "Foreign Policy & Geopolitics",
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
        "parties-media-political-narratives",
        "Parties, Media & Political Narratives",
        {"us politics", "political theory", "political history"},
        {
            "coalition",
            "democratic party",
            "disinformation",
            "media",
            "messaging",
            "narrative",
            "party",
            "polarization",
            "press",
            "republican party",
        },
    ),
]
ISSUE_AREA_RULES = [
    ("Campaign strategy and voter coalition shifts", "elections-campaigns", {"campaign", "coalition", "electability", "ticket", "voter", "voters", "swing", "strategy", "turnout", "path to victory", "map"}),
    ("Polling, debate performance and electoral momentum", "elections-campaigns", {"poll", "polls", "polling", "debate", "momentum", "undecided", "approval", "favorability", "selzer"}),
    ("Election legitimacy, voting rules and electoral systems", "elections-campaigns", {"electoral college", "mail ballot", "ballot", "voting system", "ranked choice", "election legitimacy", "certification", "third party"}),
    ("Project 2025 and the second-term governing agenda", "elections-campaigns", {"project 2025", "first 100 days", "second term", "governing agenda", "pbs", "dei"}),
    ("Supreme Court power and constitutional constraints", "institutions-law", {"supreme court", "constitutional", "constitution", "ruling", "immunity", "chevron", "judicial", "court", "courts"}),
    ("Executive power, appointments and federal governance", "institutions-law", {"executive", "attorney general", "cabinet", "agency", "senate", "congress", "governance", "appointment", "filibuster"}),
    ("Criminal cases, investigations and legal accountability", "institutions-law", {"indictment", "conviction", "trial", "judge", "jack smith", "hacked", "classified documents", "prosecution", "appeal"}),
    ("Immigration enforcement and border policy", "economy-domestic-policy", {"immigration", "border", "migrant", "asylum", "deport", "deportation"}),
    ("Inflation, housing costs and consumer pressure", "economy-domestic-policy", {"inflation", "prices", "housing", "rent", "cost of living", "groceries", "affordability"}),
    ("Labor, wages and worker bargaining power", "economy-domestic-policy", {"labor", "union", "wages", "worker", "workers", "strike", "jobs"}),
    ("Healthcare, social insurance and public spending", "economy-domestic-policy", {"healthcare", "medicare", "medicaid", "insurance", "social security", "benefits", "public option"}),
    ("Abortion rights and reproductive policy", "identity-rights-social-conflict", {"abortion", "reproductive", "roe", "pregnancy", "planned parenthood"}),
    ("Gender identity, trans rights and family policy", "identity-rights-social-conflict", {"trans", "gender", "gender-affirming", "gender affirming", "lgbt", "bathroom", "pronouns"}),
    ("Race, civil rights and protest politics", "identity-rights-social-conflict", {"black lives matter", "blm", "racial", "race", "civil rights", "police", "protest", "speech", "campus"}),
    ("Free speech, protest norms and social backlash", "identity-rights-social-conflict", {"free speech", "speech", "cancel culture", "campus", "protest", "censorship", "backlash"}),
    ("Israel, Gaza and US alignment in the Middle East", "foreign-policy-geopolitics", {"israel", "gaza", "hamas", "palestine", "middle east", "ceasefire"}),
    ("Ukraine, Russia and Western security commitments", "foreign-policy-geopolitics", {"ukraine", "russia", "putin", "nato", "europe", "aid package"}),
    ("China, trade competition and strategic rivalry", "foreign-policy-geopolitics", {"china", "taiwan", "trade", "tariff", "strategic rivalry", "manufacturing"}),
    ("Democracy, authoritarianism and institutional trust", "foreign-policy-geopolitics", {"fascist", "fascism", "dictatorship", "authoritarian", "leader free world", "democracy", "communist", "communism"}),
    ("Party identity, coalition fracture and ideological direction", "parties-media-political-narratives", {"democratic party", "republican party", "party", "coalition", "base", "establishment", "realignment", "platform"}),
    ("Media trust, messaging and narrative control", "parties-media-political-narratives", {"media", "press", "coverage", "messaging", "narrative", "disinformation", "journalist", "wikileaks"}),
    ("Vice-presidential picks, surrogates and campaign bench", "parties-media-political-narratives", {"running mate", "jd vance", "rfk", "vice president", "vp pick", "surrogate", "bench"}),
    ("MAGA identity and the Republican coalition", "parties-media-political-narratives", {"maga", "trumpism", "gop respond", "republican party", "maga movement"}),
    ("Democratic norms, political violence and institutional trust", "parties-media-political-narratives", {"assassination attempt", "political violence", "norms", "trust", "democracy", "authoritarian", "legitimacy"}),
]
NOISY_LABEL_TERMS = {
    "asked",
    "chatgpt",
    "simulate",
    "question",
    "thoughts",
    "wrong",
    "happen",
    "happened",
    "mean",
    "impact",
    "work",
    "choice",
    "https",
    "www",
    "donald",
    "president",
    "party",
    "project",
    "leader",
}
NOISY_SALIENT_PHRASES = {
    "Https Www",
    "Help Understand",
    "Look Like",
    "Make Sense",
    "Party",
    "President",
    "Donald",
    "Project",
}


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
    parser.add_argument("--disable-groq-summaries", action="store_true")
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


def clean_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" -,:;.!?")
    return text


def phrase_tokens(text: str) -> list[str]:
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


def extract_salient_phrases(topic_posts: pd.DataFrame, top_n: int = 18) -> list[str]:
    texts = (
        topic_posts["title"].fillna("")
        + ". "
        + topic_posts["selftext"].fillna("").astype(str).str.slice(0, 280)
    ).tolist()
    if not texts:
        return []
    min_df = 2 if len(texts) >= 30 else 1
    vectorizer = CountVectorizer(
        stop_words=sorted(GENERIC_TOPIC_STOPWORDS),
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=0.85,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    title_phrase_counts = Counter(score_title_phrases(topic_posts, top_n=40))
    scored: list[tuple[float, str]] = []
    total_docs = max(len(texts), 1)
    for phrase, count, freq in zip(feature_names, counts, doc_freq):
        phrase = clean_phrase(phrase)
        tokens = phrase_tokens(phrase)
        if not phrase or not tokens:
            continue
        if all(token in GENERIC_TOPIC_STOPWORDS or token in NOISY_LABEL_TERMS for token in tokens):
            continue
        if len(tokens) == 1 and freq < max(3, total_docs * 0.08):
            continue
        quality = 1.0 + 0.4 * min(len(tokens), 3)
        title_boost = 1.0 + 0.75 * title_phrase_counts.get(phrase, 0)
        specificity = 1.0 + min(freq / total_docs, 0.35)
        penalty = 0.55 if any(token in NOISY_LABEL_TERMS for token in tokens) else 1.0
        score = float(count) * quality * title_boost * specificity * penalty
        scored.append((score, phrase))
    ranked: list[str] = []
    seen: set[str] = set()
    for _, phrase in sorted(scored, reverse=True):
        normalized = phrase.lower()
        if normalized in seen:
            continue
        formatted = format_label_text(phrase)
        if formatted in NOISY_SALIENT_PHRASES:
            continue
        seen.add(normalized)
        ranked.append(formatted)
        if len(ranked) >= top_n:
            break
    return ranked


def format_label_text(text: str) -> str:
    parts = []
    for token in text.split():
        upper = token.upper()
        parts.append(upper if upper in UPPER_TOKENS else token.capitalize())
    return " ".join(parts)


def normalize_merge_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def issue_rule_domain(label: str) -> tuple[str, str] | None:
    for candidate_label, slug, _ in ISSUE_AREA_RULES:
        if candidate_label == label:
            for domain_slug, domain_name, _, _ in DOMAIN_ARCHETYPES:
                if domain_slug == slug:
                    return domain_slug, domain_name
    return None


def score_issue_area_rule(
    hint_text: str,
    title_text: str,
    flair_names: set[str],
    rule_hints: set[str],
) -> float:
    score = 0.0
    for hint in rule_hints:
        if hint in hint_text:
            score += 2.5 if " " in hint else 1.5
        if hint in title_text:
            score += 1.25
    if any(flair in flair_names for flair in ("legal/courts", "legislation")) and any(
        hint in rule_hints for hint in {"court", "courts", "law", "legal", "supreme court"}
    ):
        score += 1.5
    return score


def infer_issue_area_label(
    topic_posts: pd.DataFrame,
    keywords: list[str],
    top_flairs: list[dict[str, Any]],
    representative_titles: list[str],
    salient_phrases: list[str],
) -> str:
    candidate_phrases = [*salient_phrases, *score_title_phrases(topic_posts, top_n=20), *keywords]
    hint_text = " ".join(item.lower() for item in [*candidate_phrases, *representative_titles])
    title_text = " ".join(title.lower() for title in representative_titles)
    flair_names = {item["flair"].lower() for item in top_flairs}
    scored_rules = [
        (score_issue_area_rule(hint_text, title_text, flair_names, hints), label)
        for label, _, hints in ISSUE_AREA_RULES
    ]
    best_score, best_label = max(scored_rules, key=lambda item: item[0], default=(0.0, ""))
    if best_score >= 3.0:
        return best_label

    median_score = float(topic_posts["score"].median()) if "score" in topic_posts else 0.0
    dominant_flair = top_flairs[0]["flair"].lower() if top_flairs else ""
    if median_score <= 2:
        if dominant_flair in {"us elections", "us politics"}:
            return "General campaign chatter and voter sentiment"
        return "Low-signal general political discussion"

    for phrase in candidate_phrases:
        tokens = [token for token in phrase_tokens(phrase) if token not in GENERIC_TOPIC_STOPWORDS]
        if len(tokens) < 2:
            continue
        if any(token in NOISY_LABEL_TERMS for token in tokens):
            continue
        candidate = format_label_text(" ".join(tokens[:5]))
        if len(candidate) >= 12:
            return candidate
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


def detect_major_topic(
    topic_label: str,
    topic_keywords: list[str],
    top_flairs: list[dict[str, Any]],
    salient_phrases: list[str],
) -> tuple[str, str]:
    forced_domain = issue_rule_domain(topic_label)
    if forced_domain is not None:
        return forced_domain
    text = " ".join(
        [topic_label.lower(), *[keyword.lower() for keyword in topic_keywords], *[phrase.lower() for phrase in salient_phrases]]
    )
    flair_names = {item["flair"].lower() for item in top_flairs}
    best_slug = "parties-media-political-narratives"
    best_name = "Parties, Media & Political Narratives"
    best_score = 0
    for slug, name, flair_hints, keyword_hints in DOMAIN_ARCHETYPES:
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
    salient_phrases: list[str],
) -> str:
    focus_terms = ", ".join(salient_phrases[:3] or keywords[:4]) if (salient_phrases or keywords) else "closely related issues"
    flair_text = top_flairs[0]["flair"] if top_flairs else "mixed flairs"
    title_hints = [clean_phrase(title) for title in representative_titles[:2] if clean_phrase(title)]
    recurring_threads = ", ".join(title if len(title) <= 120 else f"{title[:117].rstrip()}..." for title in title_hints)
    if recurring_threads:
        recurring_threads = recurring_threads.rstrip(".?")
    else:
        recurring_threads = label
    return (
        f"This sub-topic sits within the broader {major_topic.lower()} domain and centers on {label.lower()}. "
        f"Across the cluster, posts repeatedly return to {focus_terms}, with the conversation most often surfacing through {flair_text.lower()} threads. "
        f"Recurring discussions include {recurring_threads}."
    )


def build_topic_description_generative(
    label: str,
    major_topic: str,
    keywords: list[str],
    salient_phrases: list[str],
    top_flairs: list[dict[str, Any]],
    representative_posts: list[dict[str, Any]],
    disable_groq_summaries: bool,
) -> str:
    fallback = build_topic_description(
        label=label,
        major_topic=major_topic,
        keywords=keywords,
        top_flairs=top_flairs,
        representative_titles=[post.get("title", "") for post in representative_posts[:3]],
        salient_phrases=salient_phrases,
    )
    if disable_groq_summaries or not groq_available():
        return fallback

    try:
        system_prompt = (
            "You are writing analytical topic descriptions for a political discussion dataset. "
            "Return strict JSON with key `topic_description`. "
            "Write 2 to 3 sentences. Be specific, synthetic, and descriptive. "
            "Describe the shared issue area tying the posts together, not just keywords or one headline. "
            "Do not mention that you are analyzing a dataset."
        )
        post_lines = "\n".join(
            f"- {post.get('title', '')} [flair={post.get('link_flair_text', 'Unspecified')}, month={post.get('created_month', '')}]"
            for post in representative_posts[:5]
        )
        flair_lines = ", ".join(f"{item['flair']} ({item['share_within_topic']:.1%})" for item in top_flairs[:4])
        payload = (
            f"Major topic domain: {major_topic}\n"
            f"Sub-topic label: {label}\n"
            f"Keywords: {', '.join(keywords[:8])}\n"
            f"Salient issue signals: {', '.join(salient_phrases[:8])}\n"
            f"Top flairs: {flair_lines}\n"
            f"Representative posts:\n{post_lines}\n\n"
            "Return JSON like {\"topic_description\": \"...\"}."
        )
        result = groq_json_completion(system_prompt, payload, max_tokens=240)
        description = clean_phrase(str(result.get("topic_description", "")))
        return description or fallback
    except Exception:
        return fallback


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
                f"- **Salient Issue Signals:** {', '.join(row.get('salient_phrases', [])[:5])}",
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
        "salient_phrases",
        "merged_topic_ids",
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


def load_existing_topic_keywords(path: Path) -> dict[int, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["topic_id"]): [str(item) for item in row.get("keywords", [])]
        for row in rows
        if int(row.get("topic_id", -1)) != -1
    }


def build_refined_topic_summary(
    posts: pd.DataFrame,
    raw_keyword_map: dict[int, list[str]],
    total_posts_by_month: pd.Series,
    disable_groq_summaries: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = posts.copy()
    posts["raw_topic_id"] = posts["topic_id"].astype(int)
    draft_rows: list[dict[str, Any]] = []
    for topic_id in sorted(topic_id for topic_id in posts["raw_topic_id"].unique() if int(topic_id) != -1):
        topic_posts = posts[posts["raw_topic_id"] == topic_id].copy()
        top_flairs = flair_breakdown(topic_posts)
        representative_title_list = summarize_titles(topic_posts, limit=3)
        raw_keywords = raw_keyword_map.get(int(topic_id), [])
        salient_phrases = extract_salient_phrases(topic_posts)
        refined_label = infer_issue_area_label(
            topic_posts=topic_posts,
            keywords=raw_keywords,
            top_flairs=top_flairs,
            representative_titles=representative_title_list,
            salient_phrases=salient_phrases,
        )
        major_topic_slug, major_topic = detect_major_topic(refined_label, raw_keywords, top_flairs, salient_phrases)
        draft_rows.append(
            {
                "raw_topic_id": int(topic_id),
                "merge_key": f"{major_topic_slug}::{normalize_merge_key(refined_label)}",
                "label": refined_label,
                "major_topic_slug": major_topic_slug,
                "major_topic": major_topic,
                "keywords": raw_keywords,
                "salient_phrases": salient_phrases,
                "top_flairs": top_flairs,
                "representative_titles": representative_title_list,
            }
        )

    refined_topics: list[dict[str, Any]] = []
    total_posts = len(posts)
    next_topic_id = 0

    for _, group in pd.DataFrame(draft_rows).groupby("merge_key", sort=False):
        raw_topic_ids = [int(value) for value in group["raw_topic_id"].tolist()]
        merged_posts = posts[posts["raw_topic_id"].isin(raw_topic_ids)].copy()
        merged_posts["topic_id"] = next_topic_id
        posts.loc[posts["raw_topic_id"].isin(raw_topic_ids), "topic_id"] = next_topic_id

        keyword_counter: Counter[str] = Counter()
        phrase_counter: Counter[str] = Counter()
        for row in group.to_dict(orient="records"):
            keyword_counter.update(row["keywords"])
            phrase_counter.update(row["salient_phrases"])
        top_flairs = flair_breakdown(merged_posts)
        representative_title_list = summarize_titles(merged_posts, limit=3)
        salient_phrases = [phrase for phrase, _ in phrase_counter.most_common(8)]
        if not salient_phrases:
            salient_phrases = extract_salient_phrases(merged_posts, top_n=8)
        keywords = [phrase for phrase, _ in keyword_counter.most_common(TOPIC_WORD_LIMIT)]
        if not keywords:
            keywords = [normalize_merge_key(phrase).replace(" ", "_") for phrase in salient_phrases[:TOPIC_WORD_LIMIT]]
        label = str(group.iloc[0]["label"])
        major_topic_slug = str(group.iloc[0]["major_topic_slug"])
        major_topic = str(group.iloc[0]["major_topic"])
        trend = classify_topic_trend(merged_posts.groupby("created_month").size(), total_posts_by_month)
        rep_posts = representative_posts(merged_posts)
        refined_topics.append(
            {
                "topic_id": next_topic_id,
                "label": label,
                "topic_description": build_topic_description_generative(
                    label=label,
                    major_topic=major_topic,
                    keywords=keywords,
                    salient_phrases=salient_phrases,
                    top_flairs=top_flairs,
                    representative_posts=rep_posts,
                    disable_groq_summaries=disable_groq_summaries,
                ),
                "major_topic_slug": major_topic_slug,
                "major_topic": major_topic,
                "keywords": keywords[:TOPIC_WORD_LIMIT],
                "salient_phrases": salient_phrases[:10],
                "post_count": int(len(merged_posts)),
                "topic_share": float(len(merged_posts) / total_posts),
                "top_flairs": top_flairs,
                "representative_titles": representative_title_list,
                "representative_posts": rep_posts,
                "merged_topic_ids": raw_topic_ids,
                **trend,
            }
        )
        next_topic_id += 1

    summary = pd.DataFrame(refined_topics).sort_values("topic_share", ascending=False).reset_index(drop=True)
    topic_id_reindex = {
        int(old_topic_id): int(new_topic_id)
        for new_topic_id, old_topic_id in enumerate(summary["topic_id"].tolist())
    }
    posts["topic_id"] = posts["topic_id"].map(topic_id_reindex).astype(int)
    summary["topic_id"] = summary["topic_id"].map(topic_id_reindex).astype(int)
    return summary.sort_values("topic_share", ascending=False).reset_index(drop=True), posts


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
        "salient_phrases",
        "merged_topic_ids",
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
        existing_post_topics = pd.read_csv(out_dir / "post_topics.csv")
        posts = posts.merge(existing_post_topics[["post_id", "topic_id"]], on="post_id", how="inner")
        raw_keyword_map = load_existing_topic_keywords(out_dir / "topic_summary.json")
        months = pd.Index(sorted(posts["created_month"].unique()), name="created_month")
        total_posts_by_month = posts.groupby("created_month").size().reindex(months, fill_value=0)
        summary, posts = build_refined_topic_summary(
            posts,
            raw_keyword_map,
            total_posts_by_month,
            args.disable_groq_summaries,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["reuse_existing"] = True
        metadata["trend_labels_refreshed"] = True
        metadata["topic_count"] = int(len(summary))
        metadata["month_axis"] = months.tolist()
        metadata["post_count"] = int(len(posts))
        metadata["assigned_non_outlier_posts"] = int(len(posts))
        metadata["outlier_posts"] = 0
        metadata["refinement_mode"] = "reuse-existing-assignments"
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
        ].to_csv(out_dir / "post_topics.csv", index=False)
        save_summary_outputs(summary, out_dir, agg_stats)
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

    raw_keyword_map = {
        int(topic_id): keyword_list(topic_model, int(topic_id), limit=TOPIC_WORD_LIMIT)
        for topic_id in sorted(non_outlier["topic_id"].unique())
    }
    summary, posts = build_refined_topic_summary(
        posts[posts["topic_id"] != -1].copy(),
        raw_keyword_map,
        total_posts_by_month,
        args.disable_groq_summaries,
    )

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
        "disable_groq_summaries": args.disable_groq_summaries,
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
