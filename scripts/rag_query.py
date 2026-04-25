"""RAG query engine for Reddit political discussion corpus.

Reads:
    data/chroma_db/   (built by build_index.py)

Requires environment variables:
    GROQ_API_KEY
    GOOGLE_API_KEY

Usage:
    python scripts/rag_query.py --question "What do users think about Biden dropping out?"
    python scripts/rag_query.py --question "..." --model groq
    python scripts/rag_query.py --question "..." --model gemini
    python scripts/rag_query.py --question "..." --model both   (default)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_CACHE_DIR = ROOT / "data" / "models" / "huggingface"

# Retrieval config
CANDIDATE_POSTS = 30          # dense retrieval pool before re-ranking
FINAL_POSTS = 5               # posts passed to LLM after diversity filter
MAX_POSTS_PER_FLAIR = 2       # diversity cap per flair category
CANDIDATE_COMMENTS = 5        # comments retrieved per selected post (query-filtered)
NO_ANSWER_THRESHOLD = 0.35    # cosine similarity below this → flag as unanswerable

# LLM model names
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyAQwnWGOmjGgdb4Jip3Rs_xR7MmrM0ci_I"

SYSTEM_PROMPT = """\
You are an assistant that answers questions about political discussions on Reddit.

Rules:
1. Answer ONLY using the Reddit excerpts provided below.
2. If the excerpts do not contain enough information to answer the question, respond \
with exactly: "The corpus does not contain sufficient information to answer this question."
3. Do not use any external knowledge or make inferences beyond what is stated in the excerpts.
4. When citing a specific claim, reference the post title in parentheses.
5. For opinion questions, attribute views to "Reddit users" or "commenters", not as objective facts.
"""

NO_ANSWER_ADDENDUM = """\

IMPORTANT: The search returned no closely matching content for this query. \
The answer is very likely not present in this corpus. \
You should clearly state that rather than speculate or use outside knowledge.
"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--model",
        choices=["groq", "gemini", "both"],
        default="both",
        help="Which LLM(s) to query",
    )
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--model-cache-dir", default=str(MODEL_CACHE_DIR))
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved chunks to stdout")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model_cache: SentenceTransformer | None = None

def get_model(cache_dir: str) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=cache_dir,
        )
    return _model_cache


def embed_query(text: str, cache_dir: str) -> list[float]:
    model = get_model(cache_dir)
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_collections(chroma_dir: str) -> tuple[chromadb.Collection, chromadb.Collection]:
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    post_collection = client.get_collection("reddit_posts")
    comment_collection = client.get_collection("reddit_comments")
    return post_collection, comment_collection


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

def cosine_sim_from_distance(distance: float) -> float:
    """
    ChromaDB cosine distance is defined as 1 - cosine_similarity.
    Convert back to similarity in [0, 1].
    """
    return max(0.0, 1.0 - distance)


def rerank_score(cosine_sim: float, reddit_score: int) -> float:
    """
    Weighted re-rank: semantic similarity × log(1 + score).
    Log dampens extreme upvote outliers. Floor score at 0.
    """
    return cosine_sim * math.log1p(max(reddit_score, 0))


def retrieve_posts(
    query_embedding: list[float],
    post_collection: chromadb.Collection,
    n_candidates: int = CANDIDATE_POSTS,
) -> tuple[list[dict[str, Any]], float]:
    """
    Step 1+2: Dense retrieval → score-weighted re-ranking.
    Returns ranked list of post dicts and the max raw cosine similarity seen.
    """
    results = post_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_candidates, post_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    posts = []
    max_cosine_sim = 0.0

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
        sim = cosine_sim_from_distance(dist)
        max_cosine_sim = max(max_cosine_sim, sim)
        ranked_score = rerank_score(sim, meta.get("score", 0))
        posts.append({
            "doc_id": doc_id,
            "chunk_text": doc_text,
            "cosine_sim": sim,
            "ranked_score": ranked_score,
            **meta,
        })

    posts.sort(key=lambda p: p["ranked_score"], reverse=True)
    return posts, max_cosine_sim


def diversity_filter(
    posts: list[dict[str, Any]],
    final_k: int = FINAL_POSTS,
    max_per_flair: int = MAX_POSTS_PER_FLAIR,
) -> list[dict[str, Any]]:
    """
    Step 3: Walk ranked posts, cap at max_per_flair per flair category.
    Returns up to final_k posts.
    """
    selected = []
    flair_counts: dict[str, int] = {}
    for post in posts:
        flair = post.get("link_flair_text") or "Unspecified"
        if flair_counts.get(flair, 0) >= max_per_flair:
            continue
        selected.append(post)
        flair_counts[flair] = flair_counts.get(flair, 0) + 1
        if len(selected) == final_k:
            break
    return selected


def retrieve_comments_for_post(
    query_embedding: list[float],
    post_id: str,
    comment_collection: chromadb.Collection,
    n: int = CANDIDATE_COMMENTS,
) -> list[dict[str, Any]]:
    """
    Step 4: For a given post_id, retrieve the top-n comments that are
    most semantically similar to the query (not just highest-scored).
    Comments were already pre-filtered to top-5-by-score at index time,
    so this retrieves from that quality-capped pool.
    """
    # Count how many comments exist for this post_id
    existing = comment_collection.get(
        where={"post_id": post_id},
        include=[],
    )
    available = len(existing["ids"])
    if available == 0:
        return []

    results = comment_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, available),
        where={"post_id": post_id},
        include=["documents", "metadatas", "distances"],
    )

    comments = []
    for doc_text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = cosine_sim_from_distance(dist)
        comments.append({
            "chunk_text": doc_text,
            "cosine_sim": sim,
            **meta,
        })

    comments.sort(key=lambda c: c["cosine_sim"], reverse=True)
    return comments


def retrieve(
    question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
) -> dict[str, Any]:
    """
    Full 4-step retrieval pipeline.
    Returns dict with keys: posts, comments_by_post, no_answer_flag.
    """
    query_embedding = embed_query(question, cache_dir)

    # Step 1+2: Dense retrieval + re-ranking
    ranked_posts, max_sim = retrieve_posts(query_embedding, post_collection)

    # Step 3: Adversarial threshold check
    no_answer_flag = max_sim < NO_ANSWER_THRESHOLD

    # Step 4: Diversity filter → final post selection
    selected_posts = diversity_filter(ranked_posts)

    # Step 5: Per-post comment retrieval (query-filtered)
    comments_by_post: dict[str, list[dict[str, Any]]] = {}
    for post in selected_posts:
        post_id = post["post_id"]
        comments_by_post[post_id] = retrieve_comments_for_post(
            query_embedding,
            post_id,
            comment_collection,
        )

    return {
        "posts": selected_posts,
        "comments_by_post": comments_by_post,
        "no_answer_flag": no_answer_flag,
        "max_cosine_sim": max_sim,
        "query_embedding": query_embedding,   # reused by callers if needed
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_context(
    posts: list[dict[str, Any]],
    comments_by_post: dict[str, list[dict[str, Any]]],
) -> str:
    """
    Render retrieved posts + their comments into a readable context block.
    """
    blocks = []
    for post in posts:
        post_id = post["post_id"]
        title = post.get("title", "")
        flair = post.get("link_flair_text") or "Unspecified"
        score = post.get("score", 0)
        month = post.get("created_month", "")
        permalink = post.get("permalink", "")
        selftext = post.get("selftext", "").strip()

        header = f"[POST] {title}"
        meta_line = f"Flair: {flair} | Score: {score} | Date: {month}"
        if permalink:
            meta_line += f" | {permalink}"

        lines = ["---", header, meta_line]
        if selftext:
            lines.append(selftext)

        post_comments = comments_by_post.get(post_id, [])
        for comment in post_comments:
            body = comment.get("body") or comment.get("chunk_text", "")
            # strip the prepended "[Post: ...]" prefix if present
            if body.startswith("[Post:"):
                body = body.split("\n", 1)[-1].strip()
            c_score = comment.get("score", 0)
            lines.append(f"\n  [COMMENT] Score: {c_score}")
            lines.append(f"  {body}")

        lines.append("---")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    context: str,
    no_answer_flag: bool,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_message) tuple.
    """
    system = SYSTEM_PROMPT
    if no_answer_flag:
        system = system + NO_ANSWER_ADDENDUM

    user = f"Reddit excerpts:\n\n{context}\n\nQuestion: {question}"
    return system, user


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def query_groq(system: str, user: str) -> str:
    """Call Groq API synchronously. Raises if GROQ_API_KEY not set."""
    from groq import Groq  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def query_gemini(system: str, user: str) -> str:
    """Call Google Gemini API synchronously. Raises if GOOGLE_API_KEY not set."""
    import google.generativeai as genai  # type: ignore

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system,
    )
    response = model.generate_content(
        user,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


async def _run_in_executor(fn, *args):
    """Run a blocking function in a thread pool so it can be awaited."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def query_both_async(
    system: str,
    user: str,
) -> dict[str, str | None]:
    """
    Call Groq and Gemini in parallel.
    Returns dict with keys 'groq' and 'gemini'.
    Errors are caught per-model so one failure doesn't block the other.
    """
    async def call_groq():
        try:
            return await _run_in_executor(query_groq, system, user)
        except Exception as exc:
            return f"[Groq error: {exc}]"

    async def call_gemini():
        try:
            return await _run_in_executor(query_gemini, system, user)
        except Exception as exc:
            return f"[Gemini error: {exc}]"

    groq_answer, gemini_answer = await asyncio.gather(call_groq(), call_gemini())
    return {"groq": groq_answer, "gemini": gemini_answer}


# ---------------------------------------------------------------------------
# Public API (used by evaluate_rag.py)
# ---------------------------------------------------------------------------

def run_rag_query(
    question: str,
    post_collection: chromadb.Collection,
    comment_collection: chromadb.Collection,
    cache_dir: str,
    model: str = "both",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Full RAG pipeline for a single question.
    Returns dict with retrieval info and LLM answers.

    Parameters
    ----------
    question    : the user's question string
    post_collection, comment_collection : open ChromaDB collections
    cache_dir   : path to HuggingFace model cache
    model       : "groq" | "gemini" | "both"
    verbose     : if True, print retrieved context to stdout

    Returns
    -------
    {
        "question": str,
        "no_answer_flag": bool,
        "max_cosine_sim": float,
        "retrieved_post_ids": list[str],
        "context": str,
        "groq_answer": str | None,
        "gemini_answer": str | None,
    }
    """
    retrieval = retrieve(question, post_collection, comment_collection, cache_dir)
    context = format_context(retrieval["posts"], retrieval["comments_by_post"])
    system, user = build_prompt(question, context, retrieval["no_answer_flag"])

    if verbose:
        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        print(context)
        print(f"\nno_answer_flag : {retrieval['no_answer_flag']}")
        print(f"max_cosine_sim : {retrieval['max_cosine_sim']:.4f}")
        print("=" * 60 + "\n")

    groq_answer = None
    gemini_answer = None

    if model in ("groq", "both"):
        if model == "groq":
            groq_answer = query_groq(system, user)
        else:
            answers = asyncio.run(query_both_async(system, user))
            groq_answer = answers["groq"]
            gemini_answer = answers["gemini"]
    elif model == "gemini":
        gemini_answer = query_gemini(system, user)

    return {
        "question": question,
        "no_answer_flag": retrieval["no_answer_flag"],
        "max_cosine_sim": retrieval["max_cosine_sim"],
        "retrieved_post_ids": [p["post_id"] for p in retrieval["posts"]],
        "context": context,
        "groq_answer": groq_answer,
        "gemini_answer": gemini_answer,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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
    )

    print("\n" + "=" * 60)
    if result["no_answer_flag"]:
        print("⚠  Low retrieval confidence — adversarial/out-of-scope query likely")
    print(f"Max cosine similarity: {result['max_cosine_sim']:.4f}")
    print(f"Retrieved posts      : {result['retrieved_post_ids']}")

    if result["groq_answer"]:
        print("\n--- Groq answer ---")
        print(result["groq_answer"])

    if result["gemini_answer"]:
        print("\n--- Gemini answer ---")
        print(result["gemini_answer"])


if __name__ == "__main__":
    main()