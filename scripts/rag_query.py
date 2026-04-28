"""RAG query engine for the Reddit political discussion corpus.

Usage examples:
    python scripts/rag_query.py --question 'focused: "What do users think about Biden dropping out?"'
    python scripts/rag_query.py --question 'aggregate: "What topics dominate the corpus?"'
    python scripts/rag_query.py --question 'comparison: "Harris vs Trump"' --model groq
    python scripts/rag_query.py --question 'multi-hop: "How did views on Harris shift after Biden dropped out?"'
    python scripts/rag_query.py --question 'aggregate: "Corpus summary?"' --lang hi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
DEFAULT_AGGREGATE_STATS_PATH = ROOT / "data" / "topic_analysis" / "aggregate_stats.json"
DEFAULT_TOPIC_SUMMARY_PATH = ROOT / "data" / "topic_analysis" / "topic_summary.json"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_CACHE_DIR = ROOT / "data" / "models" / "huggingface"

# Retrieval config
CANDIDATE_POSTS = 30
FINAL_POSTS = 5
MAX_POSTS_PER_FLAIR = 2
CANDIDATE_COMMENTS = 5
NO_ANSWER_THRESHOLD = 0.35
AGGREGATE_CANDIDATE_POSTS = 50
AGGREGATE_FINAL_POSTS = 8
AGGREGATE_MAX_POSTS_PER_FLAIR = 3
AGGREGATE_COMMENTS_PER_POST = 3
COMPARISON_POSTS_PER_SIDE = 4
MULTIHOP_POSTS_PER_HOP = 3
MAX_COMBINED_SOURCE_POSTS = 10

# LLM config
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MIN_INTERVAL_SECONDS = 5.0

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French"}
QUESTION_TYPES = ("focused", "aggregate", "comparison", "multi-hop")
QUESTION_TYPE_PREFIXES = {
    "focused": "focused:",
    "aggregate": "aggregate:",
    "comparison": "comparison:",
    "multi-hop": "multi-hop:",
}
NO_ANSWER_TEXT = "The corpus does not contain sufficient information to answer this question."

SYSTEM_PROMPT = """\
You are an assistant that answers questions about political discussions on Reddit.

Rules:
1. Use only the evidence and structured summaries provided in the prompt.
2. If the prompt does not contain enough information, respond with exactly:
   "The corpus does not contain sufficient information to answer this question."
3. Do not use outside knowledge.
4. Synthesize across multiple excerpts instead of over-weighting a single thread.
5. For opinion or stance questions, attribute views to Reddit users or commenters.
6. Always respond in English.
"""

PROMPT_MODE_GUIDANCE = {
    "focused": """\
Answer from the retrieved Reddit excerpts.
Compare recurring claims or arguments across posts and comments when possible.
Do not include inline citations or parenthetical source labels in the answer text.
""",
    "aggregate": """\
This is an aggregate question.
Use structured corpus statistics and topic summaries for corpus-level claims.
Use retrieved Reddit excerpts only as supporting examples.
If the structured aggregate data does not support the requested ranking, count, or corpus-wide conclusion, return the exact insufficient-information sentence.
Do not include inline citations or parenthetical source labels in the answer text.
""",
    "comparison": """\
This is a comparison question.
Compare the two sides explicitly and state where the evidence differs, overlaps, or is missing.
Do not imply balance if one side has materially weaker evidence.
Do not include inline citations or parenthetical source labels in the answer text.
""",
    "multi-hop": """\
This is a multi-hop question.
Only answer after connecting the evidence across the provided hops.
If one required hop is unsupported, return the exact insufficient-information sentence.
Do not include inline citations or parenthetical source labels in the answer text.
""",
}

NO_ANSWER_ADDENDUM = """\

IMPORTANT: The retrieved evidence does not confidently support this query.
If the answer is not clearly grounded in the provided material, return the exact insufficient-information sentence.
"""

_model_cache: SentenceTransformer | None = None
_aggregate_stats_cache: dict[str, Any] | None = None
_topic_summary_cache: list[dict[str, Any]] | None = None
_gemini_rate_lock = threading.Lock()
_last_gemini_call_time = 0.0


def strip_wrapping_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def parse_question_type(question: str, query_type: str = "auto") -> tuple[str, str]:
    raw = question.strip()
    detected_type = None
    parsed_question = raw
    lowered = raw.lower()

    for candidate, prefix in QUESTION_TYPE_PREFIXES.items():
        if lowered.startswith(prefix):
            detected_type = candidate
            parsed_question = raw[len(prefix):].strip()
            break

    chosen_type = detected_type or ("focused" if query_type == "auto" else query_type)
    if chosen_type not in QUESTION_TYPES:
        raise ValueError(f"Unsupported query type: {chosen_type}")
    return chosen_type, strip_wrapping_quotes(parsed_question)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_aggregate_stats(path: Path = DEFAULT_AGGREGATE_STATS_PATH) -> dict[str, Any]:
    global _aggregate_stats_cache
    if _aggregate_stats_cache is None:
        _aggregate_stats_cache = load_json_file(path)
    return _aggregate_stats_cache


def load_topic_summaries(path: Path = DEFAULT_TOPIC_SUMMARY_PATH) -> list[dict[str, Any]]:
    global _topic_summary_cache
    if _topic_summary_cache is None:
        _topic_summary_cache = load_json_file(path)
    return _topic_summary_cache


def pace_gemini_calls() -> None:
    global _last_gemini_call_time
    with _gemini_rate_lock:
        elapsed = time.monotonic() - _last_gemini_call_time
        if elapsed < GEMINI_MIN_INTERVAL_SECONDS:
            time.sleep(GEMINI_MIN_INTERVAL_SECONDS - elapsed)
        _last_gemini_call_time = time.monotonic()


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following {SUPPORTED_LANGUAGES[source_lang]} text "
                f"to English. Return only the translation.\n\n{text}"
            ),
        }],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def translate_answer(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following English text to {SUPPORTED_LANGUAGES[target_lang]}. "
                f"Preserve Reddit-specific terms and proper nouns exactly as written. "
                f"Return only the translation.\n\n{text}"
            ),
        }],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--query-type",
        choices=["auto", *QUESTION_TYPES],
        default="auto",
        help="Routing mode. 'auto' respects a prefixed question such as 'aggregate:' and otherwise defaults to focused.",
    )
    parser.add_argument(
        "--model",
        choices=["groq", "gemini", "both"],
        default="both",
        help="Which LLM(s) to query",
    )
    parser.add_argument(
        "--lang",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        default="en",
        help="Language of the input question",
    )
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--model-cache-dir", default=str(MODEL_CACHE_DIR))
    parser.add_argument("--verbose", action="store_true", help="Print retrieved context to stdout")
    return parser.parse_args()


def get_model(cache_dir: str) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache_dir)
    return _model_cache


def embed_query(text: str, cache_dir: str) -> list[float]:
    model = get_model(cache_dir)
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0].tolist()


def get_collections(chroma_dir: str) -> tuple[chromadb.Collection, chromadb.Collection]:
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("reddit_posts"), client.get_collection("reddit_comments")


def cosine_sim_from_distance(distance: float) -> float:
    return max(0.0, 1.0 - distance)


def rerank_score(cosine_sim: float, reddit_score: int) -> float:
    return cosine_sim * math.log1p(max(reddit_score, 0))


def retrieve_posts(
    query_embedding: list[float],
    post_collection: chromadb.Collection,
    n_candidates: int = CANDIDATE_POSTS,
) -> tuple[list[dict[str, Any]], float]:
    results = post_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_candidates, post_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    posts: list[dict[str, Any]] = []
    max_cosine_sim = 0.0
    for doc_id, doc_text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = cosine_sim_from_distance(dist)
        max_cosine_sim = max(max_cosine_sim, sim)
        posts.append({
            "doc_id": doc_id,
            "chunk_text": doc_text,
            "cosine_sim": sim,
            "ranked_score": rerank_score(sim, meta.get("score", 0)),
            **meta,
        })

    posts.sort(key=lambda item: item["ranked_score"], reverse=True)
    return posts, max_cosine_sim


def diversity_filter(
    posts: list[dict[str, Any]],
    final_k: int = FINAL_POSTS,
    max_per_flair: int = MAX_POSTS_PER_FLAIR,
) -> list[dict[str, Any]]:
    selected = []
    flair_counts: dict[str, int] = {}
    for post in posts:
        flair = post.get("link_flair_text") or "Unspecified"
        if flair_counts.get(flair, 0) >= max_per_flair:
            continue
        selected.append(post)
        flair_counts[flair] = flair_counts.get(flair, 0) + 1
        if len(selected) >= final_k:
            break
    return selected


def retrieve_comments_for_post(
    query_embedding: list[float],
    post_id: str,
    comment_collection: chromadb.Collection,
    n: int = CANDIDATE_COMMENTS,
) -> list[dict[str, Any]]:
    existing = comment_collection.get(where={"post_id": post_id}, include=[])
    available = len(existing["ids"])
    if available == 0:
        return []

    results = comment_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, available),
        where={"post_id": post_id},
        include=["documents", "metadatas", "distances"],
    )

    comments: list[dict[str, Any]] = []
    for doc_text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        comments.append({
            "chunk_text": doc_text,
            "cosine_sim": cosine_sim_from_distance(dist),
            **meta,
        })
    comments.sort(key=lambda item: item["cosine_sim"], reverse=True)
    return comments


def retrieve(
    question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
    n_candidates: int = CANDIDATE_POSTS,
    final_k: int = FINAL_POSTS,
    max_per_flair: int = MAX_POSTS_PER_FLAIR,
    comments_per_post: int = CANDIDATE_COMMENTS,
) -> dict[str, Any]:
    query_embedding = embed_query(question, cache_dir)
    ranked_posts, max_sim = retrieve_posts(query_embedding, post_collection, n_candidates=n_candidates)
    selected_posts = diversity_filter(ranked_posts, final_k=final_k, max_per_flair=max_per_flair)

    comments_by_post: dict[str, list[dict[str, Any]]] = {}
    for post in selected_posts:
        comments_by_post[post["post_id"]] = retrieve_comments_for_post(
            query_embedding,
            post["post_id"],
            comment_collection,
            n=comments_per_post,
        )

    return {
        "posts": selected_posts,
        "comments_by_post": comments_by_post,
        "no_answer_flag": max_sim < NO_ANSWER_THRESHOLD,
        "max_cosine_sim": max_sim,
        "query_embedding": query_embedding,
    }


def format_source_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_posts = []
    for rank, post in enumerate(posts, start=1):
        permalink = post.get("permalink", "")
        if permalink and not permalink.startswith("http"):
            permalink = "https://www.reddit.com" + permalink
        source_posts.append({
            "rank": rank,
            "post_id": post.get("post_id", ""),
            "title": post.get("title", ""),
            "flair": post.get("link_flair_text") or "Unspecified",
            "score": post.get("score", 0),
            "cosine_sim": round(post.get("cosine_sim", 0.0), 4),
            "created_month": post.get("created_month", ""),
            "permalink": permalink,
        })
    return source_posts


def render_source_posts_text(source_posts: list[dict[str, Any]]) -> str:
    if not source_posts:
        return "No source posts retrieved."

    divider = "-" * 58
    lines = ["", "Sources used (top retrieved posts)", divider]
    for post in source_posts:
        lines.append(
            f"#{post['rank']} [{post['flair']} | sim={post['cosine_sim']:.4f} | "
            f"score={post['score']:,} | {post['created_month']}]"
        )
        lines.append(f'    "{post["title"]}"')
        if post["permalink"]:
            lines.append(f"    {post['permalink']}")
        lines.append("")
    lines.append(divider)
    return "\n".join(lines)


def merge_posts(*post_lists: list[dict[str, Any]], limit: int = MAX_COMBINED_SOURCE_POSTS) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for posts in post_lists:
        for post in posts:
            post_id = str(post.get("post_id", ""))
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            merged.append(post)
            if len(merged) >= limit:
                return merged
    return merged


def format_context(posts: list[dict[str, Any]], comments_by_post: dict[str, list[dict[str, Any]]]) -> str:
    blocks = []
    for post in posts:
        post_id = post["post_id"]
        title = post.get("title", "")
        flair = post.get("link_flair_text") or "Unspecified"
        score = post.get("score", 0)
        month = post.get("created_month", "")
        permalink = post.get("permalink", "")
        selftext = post.get("selftext", "").strip()

        lines = ["---", f"[POST] {title}", f"Flair: {flair} | Score: {score} | Date: {month}"]
        if permalink:
            lines[-1] += f" | {permalink}"
        if selftext:
            lines.append(selftext)

        for comment in comments_by_post.get(post_id, []):
            body = comment.get("body") or comment.get("chunk_text", "")
            if body.startswith("[Post:"):
                body = body.split("\n", 1)[-1].strip()
            lines.append(f"\n  [COMMENT] Score: {comment.get('score', 0)}")
            lines.append(f"  {body}")

        lines.append("---")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def derive_comparison_sides(question: str) -> tuple[str, str] | None:
    patterns = [
        r"(?i)(.+?)\s+vs\.?\s+(.+)",
        r"(?i)(.+?)\s+versus\s+(.+)",
        r"(?i)difference between\s+(.+?)\s+and\s+(.+)",
        r"(?i)compare\s+(.+?)\s+(?:and|with|to)\s+(.+)",
        r"(?i)how does\s+(.+?)\s+differ from\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question.strip(" ?"))
        if match:
            left = strip_wrapping_quotes(match.group(1))
            right = strip_wrapping_quotes(match.group(2))
            if left and right:
                return left, right
    return None


def build_side_question(side: str) -> str:
    return f"What do Reddit users say about {side}?"


def derive_multihop_subqueries(question: str) -> list[str]:
    cleaned = question.strip().rstrip("?")
    subqueries = [cleaned]
    lowered = cleaned.lower()

    for connector in (" after ", " before ", " during ", " because ", " following "):
        if connector in lowered:
            idx = lowered.index(connector)
            left = cleaned[:idx].strip(" ,.")
            right = cleaned[idx + len(connector):].strip(" ,.")
            if left:
                subqueries.append(left)
            if right:
                subqueries.append(right)
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for subquery in subqueries:
        normalized = subquery.lower().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(subquery)
    return deduped[:3]


def aggregate_support_reason(question: str) -> str | None:
    lowered = question.lower()
    unsupported_keywords = (
        "politician",
        "politicians",
        "candidate",
        "candidates",
        "person",
        "people",
        "entity",
        "entities",
        "moderator",
        "moderators",
        "username",
        "usernames",
    )
    if any(keyword in lowered for keyword in unsupported_keywords):
        return (
            "The current structured aggregate layer supports corpus totals, flair frequencies, "
            "and topic-level summaries, but not canonicalized person or entity rankings."
        )
    return None


def format_aggregate_brief(question: str) -> str:
    aggregate_stats = load_aggregate_stats()
    topic_summaries = load_topic_summaries()
    top_topics = sorted(topic_summaries, key=lambda item: float(item.get("topic_share", 0.0)), reverse=True)[:6]

    lines = [
        "Structured corpus aggregates:",
        f"- Total posts: {aggregate_stats.get('total_posts', 0)}",
        f"- Total comments: {aggregate_stats.get('total_comments', 0)}",
        f"- Total unique users: {aggregate_stats.get('total_unique_users', 0)}",
        f"- Total upvotes: {aggregate_stats.get('total_upvotes', 0)}",
        f"- Date range: {aggregate_stats.get('date_range_start', '')} to {aggregate_stats.get('date_range_end', '')}",
        "",
        "Top flairs by post volume:",
    ]
    for item in aggregate_stats.get("top_flairs", [])[:6]:
        lines.append(f"- {item.get('flair', 'Unknown')}: {item.get('posts', 0)} posts")

    lines.extend(["", "Top modeled topics by share of posts:"])
    for topic in top_topics:
        lines.append(
            "- "
            f"{topic.get('label', 'Unknown topic')} "
            f"({float(topic.get('topic_share', 0.0)):.1%} of posts; "
            f"{int(topic.get('post_count', 0))} posts)"
        )
    lines.extend(["", f"Aggregate question: {question}"])
    return "\n".join(lines)


def build_prompt(
    question: str,
    context: str,
    no_answer_flag: bool,
    query_type: str = "focused",
    route_payload: dict[str, Any] | None = None,
) -> tuple[str, str]:
    system = SYSTEM_PROMPT + "\n\n" + PROMPT_MODE_GUIDANCE.get(query_type, PROMPT_MODE_GUIDANCE["focused"])
    if no_answer_flag:
        system += NO_ANSWER_ADDENDUM

    route_payload = route_payload or {}
    sections: list[str] = []

    if query_type == "aggregate":
        sections.append(route_payload.get("aggregate_brief", ""))
        unsupported_reason = route_payload.get("unsupported_reason")
        if unsupported_reason:
            sections.append(f"Aggregate support limitation:\n{unsupported_reason}")
        if context:
            sections.append(f"Supporting Reddit excerpts:\n\n{context}")
    elif query_type == "comparison":
        sections.append(
            f"Comparison side A: {route_payload.get('left_label', 'Side A')}\n\n{route_payload.get('left_context', '')}"
        )
        sections.append(
            f"Comparison side B: {route_payload.get('right_label', 'Side B')}\n\n{route_payload.get('right_context', '')}"
        )
    elif query_type == "multi-hop":
        for hop in route_payload.get("hop_contexts", []):
            note = " (low confidence)" if hop.get("no_answer_flag") else ""
            sections.append(f"{hop.get('label', 'Hop')}{note}\n\n{hop.get('context', '')}")
    else:
        sections.append(f"Reddit excerpts:\n\n{context}")

    user = "\n\n".join(section for section in sections if section).strip()
    user = f"{user}\n\nQuestion: {question}"
    return system, user


def query_groq(system: str, user: str) -> str:
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def query_gemini(system: str, user: str) -> str:
    pace_gemini_calls()
    api_key = os.environ.get("GOOGLE_API_KEY", GOOGLE_API_KEY)
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def query_both_async(system: str, user: str) -> dict[str, str | None]:
    async def call_groq():
        try:
            return await _run_in_executor(query_groq, system, user)
        except Exception as exc:  # pragma: no cover - network/runtime path
            return f"[Groq error: {exc}]"

    async def call_gemini():
        try:
            return await _run_in_executor(query_gemini, system, user)
        except Exception as exc:  # pragma: no cover - network/runtime path
            return f"[Gemini error: {exc}]"

    groq_answer, gemini_answer = await asyncio.gather(call_groq(), call_gemini())
    return {"groq": groq_answer, "gemini": gemini_answer}


def build_focused_route(
    english_question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
) -> dict[str, Any]:
    retrieval = retrieve(english_question, post_collection, comment_collection, cache_dir)
    context = format_context(retrieval["posts"], retrieval["comments_by_post"])
    system, user_msg = build_prompt(english_question, context, retrieval["no_answer_flag"], query_type="focused")
    return {
        "context": context,
        "system": system,
        "user_msg": user_msg,
        "source_posts": format_source_posts(retrieval["posts"]),
        "retrieved_post_ids": [p["post_id"] for p in retrieval["posts"]],
        "max_cosine_sim": retrieval["max_cosine_sim"],
        "no_answer_flag": retrieval["no_answer_flag"],
        "route_metadata": {},
    }


def build_aggregate_route(
    english_question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
) -> dict[str, Any]:
    retrieval = retrieve(
        english_question,
        post_collection,
        comment_collection,
        cache_dir,
        n_candidates=AGGREGATE_CANDIDATE_POSTS,
        final_k=AGGREGATE_FINAL_POSTS,
        max_per_flair=AGGREGATE_MAX_POSTS_PER_FLAIR,
        comments_per_post=AGGREGATE_COMMENTS_PER_POST,
    )
    context = format_context(retrieval["posts"], retrieval["comments_by_post"])
    unsupported_reason = aggregate_support_reason(english_question)
    no_answer_flag = retrieval["no_answer_flag"] or unsupported_reason is not None
    system, user_msg = build_prompt(
        english_question,
        context,
        no_answer_flag,
        query_type="aggregate",
        route_payload={
            "aggregate_brief": format_aggregate_brief(english_question),
            "unsupported_reason": unsupported_reason,
        },
    )
    return {
        "context": context,
        "system": system,
        "user_msg": user_msg,
        "source_posts": format_source_posts(retrieval["posts"]),
        "retrieved_post_ids": [p["post_id"] for p in retrieval["posts"]],
        "max_cosine_sim": retrieval["max_cosine_sim"],
        "no_answer_flag": no_answer_flag,
        "route_metadata": {"aggregate_unsupported_reason": unsupported_reason},
    }


def build_comparison_route(
    english_question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
) -> dict[str, Any]:
    comparison_pair = derive_comparison_sides(english_question)
    if comparison_pair is None:
        route = build_focused_route(english_question, post_collection, comment_collection, cache_dir)
        route["route_metadata"] = {"comparison_pair": None, "comparison_fallback": "focused"}
        return route

    left_label, right_label = comparison_pair
    left_retrieval = retrieve(
        build_side_question(left_label),
        post_collection,
        comment_collection,
        cache_dir,
        final_k=COMPARISON_POSTS_PER_SIDE,
        comments_per_post=AGGREGATE_COMMENTS_PER_POST,
    )
    right_retrieval = retrieve(
        build_side_question(right_label),
        post_collection,
        comment_collection,
        cache_dir,
        final_k=COMPARISON_POSTS_PER_SIDE,
        comments_per_post=AGGREGATE_COMMENTS_PER_POST,
    )
    left_context = format_context(left_retrieval["posts"], left_retrieval["comments_by_post"])
    right_context = format_context(right_retrieval["posts"], right_retrieval["comments_by_post"])
    merged_posts = merge_posts(left_retrieval["posts"], right_retrieval["posts"])
    no_answer_flag = left_retrieval["no_answer_flag"] or right_retrieval["no_answer_flag"]
    system, user_msg = build_prompt(
        english_question,
        "",
        no_answer_flag,
        query_type="comparison",
        route_payload={
            "left_label": left_label,
            "right_label": right_label,
            "left_context": left_context,
            "right_context": right_context,
        },
    )
    return {
        "context": f"Comparison side A: {left_label}\n\n{left_context}\n\nComparison side B: {right_label}\n\n{right_context}",
        "system": system,
        "user_msg": user_msg,
        "source_posts": format_source_posts(merged_posts),
        "retrieved_post_ids": [p["post_id"] for p in merged_posts],
        "max_cosine_sim": max(left_retrieval["max_cosine_sim"], right_retrieval["max_cosine_sim"]),
        "no_answer_flag": no_answer_flag,
        "route_metadata": {
            "comparison_pair": {"left": left_label, "right": right_label},
            "left_no_answer_flag": left_retrieval["no_answer_flag"],
            "right_no_answer_flag": right_retrieval["no_answer_flag"],
        },
    }


def build_multihop_route(
    english_question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
) -> dict[str, Any]:
    hop_results = []
    for index, hop_question in enumerate(derive_multihop_subqueries(english_question), start=1):
        retrieval = retrieve(
            hop_question,
            post_collection,
            comment_collection,
            cache_dir,
            final_k=MULTIHOP_POSTS_PER_HOP,
            comments_per_post=AGGREGATE_COMMENTS_PER_POST,
        )
        hop_results.append({
            "question": hop_question,
            "label": f"Hop {index}: {hop_question}",
            "context": format_context(retrieval["posts"], retrieval["comments_by_post"]),
            "posts": retrieval["posts"],
            "max_cosine_sim": retrieval["max_cosine_sim"],
            "no_answer_flag": retrieval["no_answer_flag"],
        })

    merged_posts = merge_posts(*[hop["posts"] for hop in hop_results])
    no_answer_flag = any(hop["no_answer_flag"] for hop in hop_results)
    system, user_msg = build_prompt(
        english_question,
        "",
        no_answer_flag,
        query_type="multi-hop",
        route_payload={"hop_contexts": hop_results},
    )
    return {
        "context": "\n\n".join(f"{hop['label']}\n\n{hop['context']}" for hop in hop_results),
        "system": system,
        "user_msg": user_msg,
        "source_posts": format_source_posts(merged_posts),
        "retrieved_post_ids": [p["post_id"] for p in merged_posts],
        "max_cosine_sim": max((hop["max_cosine_sim"] for hop in hop_results), default=0.0),
        "no_answer_flag": no_answer_flag,
        "route_metadata": {
            "hop_questions": [hop["question"] for hop in hop_results],
            "hop_no_answer_flags": [hop["no_answer_flag"] for hop in hop_results],
        },
    }


def run_rag_query(
    question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
    model: str = "both",
    verbose: bool = False,
    lang: str = "en",
    query_type: str = "auto",
) -> dict[str, Any]:
    resolved_query_type, question_body = parse_question_type(question, query_type)
    english_question = translate_to_english(question_body, lang)

    if resolved_query_type == "aggregate":
        route_result = build_aggregate_route(english_question, post_collection, comment_collection, cache_dir)
    elif resolved_query_type == "comparison":
        route_result = build_comparison_route(english_question, post_collection, comment_collection, cache_dir)
    elif resolved_query_type == "multi-hop":
        route_result = build_multihop_route(english_question, post_collection, comment_collection, cache_dir)
    else:
        route_result = build_focused_route(english_question, post_collection, comment_collection, cache_dir)

    if verbose:
        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        print(route_result["context"])
        print(f"\nquery_type     : {resolved_query_type}")
        print(f"no_answer_flag : {route_result['no_answer_flag']}")
        print(f"max_cosine_sim : {route_result['max_cosine_sim']:.4f}")
        if lang != "en":
            print(f"original question ({lang}): {question_body}")
            print(f"translated to (en)        : {english_question}")
        print("=" * 60 + "\n")
        print(render_source_posts_text(route_result["source_posts"]))

    groq_answer: str | None = None
    gemini_answer: str | None = None
    if model in ("groq", "both"):
        if model == "groq":
            groq_answer = query_groq(route_result["system"], route_result["user_msg"])
        else:
            answers = asyncio.run(query_both_async(route_result["system"], route_result["user_msg"]))
            groq_answer = answers["groq"]
            gemini_answer = answers["gemini"]
    elif model == "gemini":
        gemini_answer = query_gemini(route_result["system"], route_result["user_msg"])

    if lang != "en":
        if groq_answer:
            groq_answer = translate_answer(groq_answer, lang)
        if gemini_answer:
            gemini_answer = translate_answer(gemini_answer, lang)

    return {
        "question": question,
        "question_body": question_body,
        "query_type": resolved_query_type,
        "english_question": english_question,
        "lang": lang,
        "no_answer_flag": route_result["no_answer_flag"],
        "max_cosine_sim": route_result["max_cosine_sim"],
        "retrieved_post_ids": route_result["retrieved_post_ids"],
        "source_posts": route_result["source_posts"],
        "context": route_result["context"],
        "route_metadata": route_result["route_metadata"],
        "groq_answer": groq_answer,
        "gemini_answer": gemini_answer,
    }


def main() -> None:
    args = parse_args()
    post_collection, comment_collection = get_collections(args.chroma_dir)
    result = run_rag_query(
        question=args.question,
        post_collection=post_collection,
        comment_collection=comment_collection,
        cache_dir=args.model_cache_dir,
        model=args.model,
        verbose=args.verbose,
        lang=args.lang,
        query_type=args.query_type,
    )

    print("\n" + "=" * 60)
    if result["no_answer_flag"]:
        print("Low retrieval confidence or unsupported routed query.")
    print(f"Query type           : {result['query_type']}")
    print(f"Language             : {SUPPORTED_LANGUAGES[result['lang']]}")
    if result["lang"] != "en":
        print(f"Original question    : {result['question_body']}")
        print(f"Translated to English: {result['english_question']}")
    print(f"Max cosine similarity: {result['max_cosine_sim']:.4f}")
    print(render_source_posts_text(result["source_posts"]))

    if result["groq_answer"]:
        print("\n--- Groq answer ---")
        print(result["groq_answer"])
    if result["gemini_answer"]:
        print("\n--- Gemini answer ---")
        print(result["gemini_answer"])


if __name__ == "__main__":
    main()
