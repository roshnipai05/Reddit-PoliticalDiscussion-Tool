"""Clean Reddit JSONL exports for downstream NLP tasks.

The script preserves the raw exports and writes smaller, analysis-oriented
JSONL files with deleted content, moderator comments, and unneeded fields
removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DELETED_MARKERS = {
    "",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
    "[ Removed by Reddit ]".lower(),
}

MODERATOR_AUTHORS = {"automoderator"}
MODERATOR_DISTINGUISHED = {"moderator", "admin", "special"}
MOJIBAKE_HINTS = ("â", "Â", "€", "™")
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_HASH_SALT = "reddit-topic-analysis-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posts",
        default="r_parenting_posts-6-months.jsonl",
        help="Path to raw posts JSONL.",
    )
    parser.add_argument(
        "--comments",
        default="r_parenting_comments-6-months.jsonl",
        help="Path to raw comments JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/cleaned",
        help="Directory for cleaned outputs.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Drop posts/comments whose normalized text is shorter than this.",
    )
    parser.add_argument(
        "--hash-salt",
        default=DEFAULT_HASH_SALT,
        help="Salt used for deterministic author hashes.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def write_jsonl(handle, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def repair_mojibake(value: str) -> str:
    """Best-effort fix for UTF-8 text decoded as Windows-1252/Latin-1."""
    if not any(hint in value for hint in MOJIBAKE_HINTS):
        return value
    for encoding in ("cp1252", "latin-1"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if repaired.count("�") <= value.count("�"):
            return repaired
    return value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = repair_mojibake(str(value))
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def is_deleted_text(value: Any) -> bool:
    return normalize_text(value).lower() in DELETED_MARKERS


def utc_iso(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    try:
        seconds = float(timestamp)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def author_hash(author: Any, salt: str) -> str | None:
    author_text = normalize_text(author)
    if not author_text or author_text.lower() == "[deleted]":
        return None
    digest = hashlib.sha256(f"{salt}:{author_text}".encode("utf-8")).hexdigest()
    return digest[:16]


def clean_post(raw: dict[str, Any], salt: str, min_text_chars: int) -> tuple[dict[str, Any] | None, str | None]:
    title = normalize_text(raw.get("title"))
    selftext = normalize_text(raw.get("selftext"))

    if is_deleted_text(raw.get("selftext")):
        return None, "empty_deleted_or_removed_selftext"

    combined_text = f"{title}\n\n{selftext}".strip() if selftext else title
    if len(combined_text) < min_text_chars:
        return None, "too_short"

    post_id = normalize_text(raw.get("id"))
    if not post_id:
        return None, "missing_id"

    cleaned = {
        "post_id": post_id,
        "created_utc": raw.get("created_utc"),
        "created_iso": utc_iso(raw.get("created_utc")),
        "subreddit": normalize_text(raw.get("subreddit")),
        "author_hash": author_hash(raw.get("author"), salt),
        "title": title,
        "selftext": selftext,
        "text": combined_text,
        "link_flair_text": normalize_text(raw.get("link_flair_text")),
        "score": raw.get("score"),
        "upvote_ratio": raw.get("upvote_ratio"),
        "num_comments": raw.get("num_comments"),
        "over_18": bool(raw.get("over_18")),
        "permalink": normalize_text(raw.get("permalink")),
        "url": normalize_text(raw.get("url")),
    }
    return cleaned, None


def is_moderator_comment(raw: dict[str, Any]) -> bool:
    author = normalize_text(raw.get("author")).lower()
    distinguished = normalize_text(raw.get("distinguished")).lower()
    return author in MODERATOR_AUTHORS or distinguished in MODERATOR_DISTINGUISHED


def clean_comment(
    raw: dict[str, Any],
    salt: str,
    min_text_chars: int,
    retained_posts: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if is_moderator_comment(raw):
        return None, "moderator_or_automoderator"

    body = normalize_text(raw.get("body"))
    if is_deleted_text(body):
        return None, "empty_deleted_or_removed_body"
    if len(body) < min_text_chars:
        return None, "too_short"

    comment_id = normalize_text(raw.get("id"))
    if not comment_id:
        return None, "missing_id"

    post_id = normalize_text(raw.get("link_id")).removeprefix("t3_")
    if not post_id:
        return None, "missing_post_id"
    if post_id not in retained_posts:
        return None, "post_not_retained"

    cleaned = {
        "comment_id": comment_id,
        "post_id": post_id,
        "parent_id": normalize_text(raw.get("parent_id")),
        "created_utc": raw.get("created_utc"),
        "created_iso": utc_iso(raw.get("created_utc")),
        "subreddit": normalize_text(raw.get("subreddit")),
        "author_hash": author_hash(raw.get("author"), salt),
        "body": body,
        "text": body,
        "score": raw.get("score"),
        "is_submitter": bool(raw.get("is_submitter")),
        "permalink": normalize_text(raw.get("permalink")),
    }
    return cleaned, None


def clean_posts(posts_path: Path, out_path: Path, salt: str, min_text_chars: int):
    stats = Counter()
    retained_posts: set[str] = set()

    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for _, raw in iter_jsonl(posts_path):
            stats["raw"] += 1
            cleaned, reason = clean_post(raw, salt, min_text_chars)
            if reason:
                stats[f"dropped_{reason}"] += 1
                continue
            assert cleaned is not None
            retained_posts.add(cleaned["post_id"])
            write_jsonl(out_handle, cleaned)
            stats["retained"] += 1

    return stats, retained_posts


def clean_comments(
    comments_path: Path,
    out_path: Path,
    salt: str,
    min_text_chars: int,
    retained_posts: set[str],
):
    stats = Counter()

    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for _, raw in iter_jsonl(comments_path):
            stats["raw"] += 1
            cleaned, reason = clean_comment(raw, salt, min_text_chars, retained_posts)
            if reason:
                stats[f"dropped_{reason}"] += 1
                continue
            assert cleaned is not None
            write_jsonl(out_handle, cleaned)
            stats["retained"] += 1

    return stats


def write_documents(posts_path: Path, comments_path: Path, out_path: Path) -> Counter:
    stats = Counter()
    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for _, post in iter_jsonl(posts_path):
            document = {
                "doc_id": f"post_{post['post_id']}",
                "doc_type": "post",
                "post_id": post["post_id"],
                "created_utc": post["created_utc"],
                "created_iso": post["created_iso"],
                "author_hash": post["author_hash"],
                "title": post["title"],
                "text": post["text"],
                "score": post["score"],
                "permalink": post["permalink"],
                "link_flair_text": post["link_flair_text"],
            }
            write_jsonl(out_handle, document)
            stats["posts"] += 1

        for _, comment in iter_jsonl(comments_path):
            document = {
                "doc_id": f"comment_{comment['comment_id']}",
                "doc_type": "comment",
                "comment_id": comment["comment_id"],
                "post_id": comment["post_id"],
                "parent_id": comment["parent_id"],
                "created_utc": comment["created_utc"],
                "created_iso": comment["created_iso"],
                "author_hash": comment["author_hash"],
                "text": comment["text"],
                "score": comment["score"],
                "permalink": comment["permalink"],
            }
            write_jsonl(out_handle, document)
            stats["comments"] += 1
    stats["total"] = stats["posts"] + stats["comments"]
    return stats


def main() -> None:
    args = parse_args()
    posts_path = Path(args.posts)
    comments_path = Path(args.comments)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_posts_path = out_dir / "posts_clean.jsonl"
    cleaned_comments_path = out_dir / "comments_clean.jsonl"
    documents_path = out_dir / "documents_clean.jsonl"
    summary_path = out_dir / "cleaning_summary.json"

    post_stats, retained_posts = clean_posts(
        posts_path,
        cleaned_posts_path,
        args.hash_salt,
        args.min_text_chars,
    )
    comment_stats = clean_comments(
        comments_path,
        cleaned_comments_path,
        args.hash_salt,
        args.min_text_chars,
        retained_posts,
    )
    document_stats = write_documents(cleaned_posts_path, cleaned_comments_path, documents_path)

    summary = {
        "inputs": {
            "posts": str(posts_path),
            "comments": str(comments_path),
        },
        "outputs": {
            "posts": str(cleaned_posts_path),
            "comments": str(cleaned_comments_path),
            "documents": str(documents_path),
            "summary": str(summary_path),
        },
        "filters": {
            "deleted_post_selftext_markers": sorted(DELETED_MARKERS),
            "deleted_comment_body_markers": sorted(DELETED_MARKERS),
            "moderator_comment_authors": sorted(MODERATOR_AUTHORS),
            "moderator_comment_distinguished_values": sorted(MODERATOR_DISTINGUISHED),
            "min_text_chars": args.min_text_chars,
            "comments_restricted_to_retained_posts": True,
            "authors_hashed": True,
            "mojibake_repair": "best_effort_cp1252_or_latin1_to_utf8",
        },
        "posts": dict(post_stats),
        "comments": dict(comment_stats),
        "documents": dict(document_stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
