"""Build a compact JSON bundle for the local inspection app."""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
APP_DIR = ROOT / "app"
STANCE_DIR_CANDIDATES = [
    DATA_DIR / "topic_stance",
    DATA_DIR / "topic_stance_preview",
]
KNOWN_POLITICAL_EVENTS = [
    {"date": "2024-06-27", "month": "2024-06", "label": "First presidential debate"},
    {"date": "2024-06-28", "month": "2024-06", "label": "Chevron deference overturned"},
    {"date": "2024-07-13", "month": "2024-07", "label": "Trump assassination attempt"},
    {"date": "2024-07-21", "month": "2024-07", "label": "Biden exits race"},
    {"date": "2024-08-19", "month": "2024-08", "label": "Democratic National Convention"},
    {"date": "2024-09-10", "month": "2024-09", "label": "Harris-Trump debate"},
    {"date": "2024-11-05", "month": "2024-11", "label": "US Election Day"},
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path, default: Any) -> Any:
    if path.exists():
        return load_json(path)
    return default


def resolve_stance_dir() -> Path:
    for candidate in STANCE_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return STANCE_DIR_CANDIDATES[0]


def load_comment_preview(path: Path, per_topic: int = 12) -> dict[int, list[dict[str, Any]]]:
    preview: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return preview
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            topic_id = int(row["topic_id"])
            preview.setdefault(topic_id, [])
            if len(preview[topic_id]) >= per_topic:
                continue
            preview[topic_id].append(
                {
                    "comment_id": row["comment_id"],
                    "post_id": row["post_id"],
                    "topic_label": row["topic_label"],
                    "author_hash": row["author_hash"],
                    "created_iso": row["created_iso"],
                    "score": int(row["score"]),
                    "stance_label": row["stance_label"],
                    "stance_confidence": float(row["stance_confidence"]),
                    "body": row["body"],
                    "permalink": row["permalink"],
                    "topic_post_title": row["topic_post_title"],
                    "topic_post_permalink": row["topic_post_permalink"],
                    "link_flair_text": row["link_flair_text"],
                }
            )
    return preview


def load_daily_topic_timelines() -> tuple[list[str], dict[int, list[int]]]:
    post_topics_path = DATA_DIR / "topic_analysis" / "post_topics.csv"
    posts_path = DATA_DIR / "cleaned" / "posts_clean.jsonl"
    if not post_topics_path.exists() or not posts_path.exists():
        return [], {}

    post_to_topic: dict[str, int] = {}
    with post_topics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            post_id = row.get("post_id")
            topic_id = row.get("topic_id")
            if not post_id or topic_id is None:
                continue
            post_to_topic[post_id] = int(topic_id)

    counts_by_topic: dict[int, dict[str, int]] = {}
    min_day: date | None = None
    max_day: date | None = None
    with posts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            post_id = row.get("post_id")
            topic_id = post_to_topic.get(post_id)
            created_iso = row.get("created_iso")
            if topic_id is None or not created_iso:
                continue
            day = str(created_iso)[:10]
            parsed_day = date.fromisoformat(day)
            if min_day is None or parsed_day < min_day:
                min_day = parsed_day
            if max_day is None or parsed_day > max_day:
                max_day = parsed_day
            counts_by_topic.setdefault(topic_id, {})
            counts_by_topic[topic_id][day] = counts_by_topic[topic_id].get(day, 0) + 1

    if min_day is None or max_day is None:
        return [], {}

    day_axis: list[str] = []
    cursor = min_day
    while cursor <= max_day:
        day_axis.append(cursor.isoformat())
        cursor += timedelta(days=1)

    aligned_counts = {
        topic_id: [day_counts.get(day, 0) for day in day_axis]
        for topic_id, day_counts in counts_by_topic.items()
    }
    return day_axis, aligned_counts


def build_topic_tree(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for topic in topics:
        slug = topic["major_topic_slug"]
        node = grouped.setdefault(
            slug,
            {
                "id": slug,
                "label": topic["major_topic"],
                "post_count": 0,
                "topic_share": 0.0,
                "children": [],
            },
        )
        node["post_count"] += int(topic["post_count"])
        node["topic_share"] += float(topic["topic_share"])
        node["children"].append(
            {
                "id": str(topic["topic_id"]),
                "topic_id": int(topic["topic_id"]),
                "label": topic["label"],
                "description": topic["topic_description"],
                "topic_share": float(topic["topic_share"]),
                "post_count": int(topic["post_count"]),
                "trend_type": topic["trend_type"],
            }
        )
    for node in grouped.values():
        node["children"].sort(key=lambda item: item["topic_share"], reverse=True)
    return sorted(grouped.values(), key=lambda item: item["topic_share"], reverse=True)


def build_bundle() -> dict[str, Any]:
    aggregate = load_json(DATA_DIR / "topic_analysis" / "aggregate_stats.json")
    topic_run = load_json(DATA_DIR / "topic_analysis" / "run_metadata.json")
    topics = load_json(DATA_DIR / "topic_analysis" / "topic_summary.json")
    stance_dir = resolve_stance_dir()
    stance_preview = load_json_if_exists(stance_dir / "topic_stance_summary.json", [])
    stance_meta = load_json_if_exists(
        stance_dir / "run_metadata.json",
        {
            "output_dir": str(stance_dir),
            "comment_count_analyzed": 0,
            "topic_count_analyzed": 0,
            "status": "missing",
        },
    )
    user_groups = load_json_if_exists(stance_dir / "topic_user_stance_groups.json", [])
    comment_preview = load_comment_preview(stance_dir / "comment_stances.csv")

    stance_by_topic = {int(item["topic_id"]): item for item in stance_preview}
    user_groups_by_topic = {int(item["topic_id"]): item for item in user_groups}
    month_axis = topic_run["month_axis"]
    day_axis, daily_counts_by_topic = load_daily_topic_timelines()
    day_axis_lookup = set(day_axis)
    events_in_range = [
        event for event in KNOWN_POLITICAL_EVENTS if event["month"] in month_axis or event["date"] in day_axis_lookup
    ]

    enriched_topics: list[dict[str, Any]] = []
    trend_counts = {"Persistent": 0, "Trending": 0, "Declining": 0, "Episodic": 0}

    for topic in topics:
        topic_id = int(topic["topic_id"])
        trend_counts[topic["trend_type"]] = trend_counts.get(topic["trend_type"], 0) + 1
        enriched_topics.append(
            {
                **topic,
                "stance_preview": stance_by_topic.get(topic_id),
                "user_groups_preview": user_groups_by_topic.get(topic_id),
                "comment_preview": comment_preview.get(topic_id, []),
                "timeline": {
                    "months": month_axis,
                    "post_counts": topic.get("monthly_post_count_series", []),
                    "post_shares": topic.get("monthly_share_series", []),
                    "day_axis": day_axis,
                    "daily_post_counts": daily_counts_by_topic.get(topic_id, []),
                    "events": events_in_range,
                },
            }
        )

    subreddit_name = "r/Unknown"
    if enriched_topics:
        top_permalink = enriched_topics[0].get("representative_posts", [{}])[0].get("permalink", "")
        if top_permalink.startswith("/r/"):
            parts = top_permalink.strip("/").split("/")
            # permalink format: /r/<subreddit>/comments/...
            # after strip+split: parts[0]='r', parts[1]='<subreddit>'
            subreddit_name = f"{parts[0]}/{parts[1]}"

    return {
        "app_meta": {
            "title": "Political Discussion Analysis Explorer",
            "subreddit": subreddit_name,
            "analysis_scope": "Project Part 1 tasks 1.1-1.4",
            "stance_mode": "topic_level_preview" if stance_preview else "unavailable",
            "month_axis": month_axis,
            "events": events_in_range,
        },
        "aggregate_stats": aggregate,
        "topic_run_metadata": topic_run,
        "stance_preview_metadata": stance_meta,
        "overview": {
            "topic_count": len(topics),
            "persistent_topics": trend_counts.get("Persistent", 0),
            "trending_topics": trend_counts.get("Trending", 0),
            "declining_topics": trend_counts.get("Declining", 0),
            "episodic_topics": trend_counts.get("Episodic", 0),
            "stance_preview_topics": len(stance_preview),
            "stance_preview_comments": stance_meta["comment_count_analyzed"],
        },
        "topic_tree": build_topic_tree(enriched_topics),
        "topics": enriched_topics,
    }


def main() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    out_path = APP_DIR / "data.bundle.json"
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
