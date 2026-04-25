"""Build ChromaDB vector index from cleaned Reddit posts and comments.

Reads:
    data/cleaned/posts_clean.jsonl
    data/cleaned/comments_clean.jsonl

Writes:
    data/chroma_db/   (ChromaDB PersistentClient directory)
        reddit_posts collection
        reddit_comments collection
    data/chroma_db/index_metadata.json   (build stats and config)

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --posts data/cleaned/posts_clean.jsonl
    python scripts/build_index.py --reset   # drop and rebuild collections
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTS_PATH = ROOT / "data" / "cleaned" / "posts_clean.jsonl"
DEFAULT_COMMENTS_PATH = ROOT / "data" / "cleaned" / "comments_clean.jsonl"
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_CACHE_DIR = ROOT / "data" / "models" / "huggingface"
TOP_COMMENTS_PER_POST = 5
BATCH_SIZE = 256   # ChromaDB upsert batch size


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", default=str(DEFAULT_POSTS_PATH),
                        help="Path to posts_clean.jsonl")
    parser.add_argument("--comments", default=str(DEFAULT_COMMENTS_PATH),
                        help="Path to comments_clean.jsonl")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR),
                        help="ChromaDB persistence directory")
    parser.add_argument("--model-cache-dir", default=str(MODEL_CACHE_DIR),
                        help="HuggingFace model cache directory")
    parser.add_argument("--reset", action="store_true",
                        help="Delete and rebuild collections from scratch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Upsert batch size for ChromaDB")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Post loading and chunk construction
# ---------------------------------------------------------------------------

def load_posts(path: Path) -> list[dict[str, Any]]:
    """Load all posts and build chunk text + metadata."""
    posts = []
    for raw in iter_jsonl(path):
        post_id = raw.get("post_id", "").strip()
        title = raw.get("title", "").strip()
        if not post_id or not title:
            continue

        selftext = raw.get("selftext", "") or ""
        selftext = selftext.strip()
        flair = raw.get("link_flair_text") or "Unspecified"

        # Chunk text: flair tag + title + selftext body
        if selftext:
            chunk_text = f"[{flair}] {title}\n\n{selftext}"
        else:
            chunk_text = f"[{flair}] {title}"

        # created_month from created_iso; fall back to created_utc
        created_iso = raw.get("created_iso") or ""
        created_month = created_iso[:7] if created_iso else ""

        # score: some posts have None; default to 0
        score = raw.get("score")
        score = int(score) if score is not None else 0

        num_comments = raw.get("num_comments")
        num_comments = int(num_comments) if num_comments is not None else 0

        permalink = raw.get("permalink") or ""
        if permalink and not permalink.startswith("http"):
            permalink = "https://www.reddit.com" + permalink

        posts.append({
            "doc_id": f"post_{post_id}",
            "chunk_text": chunk_text,
            # metadata stored in ChromaDB (all values must be str/int/float/bool)
            "meta": {
                "doc_type": "post",
                "post_id": post_id,
                "title": title,
                "selftext": selftext[:500],   # truncated for metadata; full text in chunk
                "created_month": created_month,
                "created_iso": created_iso,
                "score": score,
                "num_comments": num_comments,
                "link_flair_text": str(flair),
                "permalink": permalink,
                "subreddit": raw.get("subreddit") or "",
            },
        })
    return posts


# ---------------------------------------------------------------------------
# Comment loading and chunk construction
# ---------------------------------------------------------------------------

def load_top_comments(path: Path, top_n: int = TOP_COMMENTS_PER_POST) -> list[dict[str, Any]]:
    """
    Load comments, group by post_id, keep top_n by score per post.
    Returns flat list of comment chunk dicts (chunk_text includes parent title placeholder;
    parent titles are resolved after posts are loaded).
    """
    # Group by post_id, keeping top_n by score
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in iter_jsonl(path):
        comment_id = raw.get("comment_id", "").strip()
        post_id = raw.get("post_id", "").strip()
        body = raw.get("body") or raw.get("text") or ""
        body = body.strip()
        if not comment_id or not post_id or not body:
            continue

        score = raw.get("score")
        score = int(score) if score is not None else 0

        created_iso = raw.get("created_iso") or ""

        permalink = raw.get("permalink") or ""
        if permalink and not permalink.startswith("http"):
            permalink = "https://www.reddit.com" + permalink

        grouped[post_id].append({
            "comment_id": comment_id,
            "post_id": post_id,
            "body": body,
            "score": score,
            "created_iso": created_iso,
            "permalink": permalink,
            "author_hash": raw.get("author_hash") or "",
        })

    # Keep top_n per post by score
    comments = []
    for post_id, post_comments in grouped.items():
        top = sorted(post_comments, key=lambda c: c["score"], reverse=True)[:top_n]
        comments.extend(top)

    return comments


def attach_parent_titles(
    comments: list[dict[str, Any]],
    posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build final comment chunk dicts with parent post title prepended.
    Comments whose parent post is not in the posts list are dropped.
    """
    title_lookup: dict[str, str] = {p["meta"]["post_id"]: p["meta"]["title"] for p in posts}

    result = []
    for c in comments:
        post_id = c["post_id"]
        parent_title = title_lookup.get(post_id)
        if not parent_title:
            continue   # parent post was filtered out — skip comment

        chunk_text = f"[Post: {parent_title}]\n{c['body']}"

        result.append({
            "doc_id": f"comment_{c['comment_id']}",
            "chunk_text": chunk_text,
            "meta": {
                "doc_type": "comment",
                "comment_id": c["comment_id"],
                "post_id": post_id,
                "parent_post_title": parent_title,
                "body": c["body"][:500],   # truncated for metadata storage
                "score": c["score"],
                "created_iso": c["created_iso"],
                "permalink": c["permalink"],
            },
        })
    return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model(model_name: str, cache_dir: Path) -> SentenceTransformer:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir),
    )


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    desc: str = "Embedding",
) -> list[list[float]]:
    """Embed in batches with progress bar, return list of float vectors."""
    all_embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(vecs.tolist())
    return all_embeddings


# ---------------------------------------------------------------------------
# ChromaDB upsert
# ---------------------------------------------------------------------------

def upsert_collection(
    collection: chromadb.Collection,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int,
) -> None:
    """Upsert chunks + embeddings into a ChromaDB collection in batches."""
    total = len(chunks)
    for start in tqdm(range(0, total, batch_size), desc=f"Upserting {collection.name}"):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]
        collection.upsert(
            ids=[c["doc_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["chunk_text"] for c in batch_chunks],
            metadatas=[c["meta"] for c in batch_chunks],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    posts_path = Path(args.posts)
    comments_path = Path(args.comments)
    chroma_dir = Path(args.chroma_dir)
    model_cache_dir = Path(args.model_cache_dir)

    for p in (posts_path, comments_path):
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    chroma_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and prepare chunks ---
    print("Loading posts...")
    posts = load_posts(posts_path)
    print(f"  {len(posts):,} posts loaded")

    print("Loading comments (top-5 per post by score)...")
    raw_comments = load_top_comments(comments_path, top_n=TOP_COMMENTS_PER_POST)
    comments = attach_parent_titles(raw_comments, posts)
    print(f"  {len(comments):,} comments loaded")

    # --- Load embedding model ---
    model = load_model(EMBEDDING_MODEL, model_cache_dir)

    # --- Embed ---
    post_texts = [p["chunk_text"] for p in posts]
    comment_texts = [c["chunk_text"] for c in comments]

    post_embeddings = embed_texts(model, post_texts, desc="Embedding posts")
    comment_embeddings = embed_texts(model, comment_texts, desc="Embedding comments")

    # --- ChromaDB setup ---
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    if args.reset:
        print("--reset: deleting existing collections...")
        for name in ("reddit_posts", "reddit_comments"):
            try:
                client.delete_collection(name)
                print(f"  Deleted collection: {name}")
            except Exception:
                pass   # collection didn't exist yet

    post_collection = client.get_or_create_collection(
        name="reddit_posts",
        metadata={"hnsw:space": "cosine"},
    )
    comment_collection = client.get_or_create_collection(
        name="reddit_comments",
        metadata={"hnsw:space": "cosine"},
    )

    # --- Upsert ---
    upsert_collection(post_collection, posts, post_embeddings, args.batch_size)
    upsert_collection(comment_collection, comments, comment_embeddings, args.batch_size)

    # --- Verify and save metadata ---
    final_post_count = post_collection.count()
    final_comment_count = comment_collection.count()
    print(f"\nIndex built successfully.")
    print(f"  reddit_posts   : {final_post_count:,} documents")
    print(f"  reddit_comments: {final_comment_count:,} documents")

    index_meta = {
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "posts_source": str(posts_path),
        "comments_source": str(comments_path),
        "chroma_dir": str(chroma_dir),
        "top_comments_per_post": TOP_COMMENTS_PER_POST,
        "post_count": final_post_count,
        "comment_count": final_comment_count,
        "collections": ["reddit_posts", "reddit_comments"],
    }
    meta_path = chroma_dir / "index_metadata.json"
    meta_path.write_text(json.dumps(index_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()