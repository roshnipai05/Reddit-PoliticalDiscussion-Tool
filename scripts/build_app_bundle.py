"""Build a compact JSON bundle for the local inspection app."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
APP_DIR = ROOT / "app"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_comment_preview(path: Path, per_topic: int = 20) -> dict[int, list[dict[str, Any]]]:
    preview: dict[int, list[dict[str, Any]]] = {}
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


def build_bundle() -> dict[str, Any]:
    aggregate = load_json(DATA_DIR / "topic_analysis" / "aggregate_stats.json")
    topic_run = load_json(DATA_DIR / "topic_analysis" / "run_metadata.json")
    topics = load_json(DATA_DIR / "topic_analysis" / "topic_summary.json")
    stance_preview = load_json(DATA_DIR / "topic_stance_preview" / "topic_stance_summary.json")
    stance_meta = load_json(DATA_DIR / "topic_stance_preview" / "run_metadata.json")
    user_groups = load_json(DATA_DIR / "topic_stance_preview" / "topic_user_stance_groups.json")
    comment_preview = load_comment_preview(DATA_DIR / "topic_stance_preview" / "comment_stances.csv")

    stance_by_topic = {int(item["topic_id"]): item for item in stance_preview}
    user_groups_by_topic = {int(item["topic_id"]): item for item in user_groups}

    enriched_topics: list[dict[str, Any]] = []
    persistent_count = 0
    trending_count = 0
    episodic_count = 0

    for topic in topics:
        topic_id = int(topic["topic_id"])
        trend = topic["trend_type"].lower()
        if trend == "persistent":
            persistent_count += 1
        elif trend == "trending":
            trending_count += 1
        elif trend == "episodic":
            episodic_count += 1

        enriched_topics.append(
            {
                **topic,
                "stance_preview": stance_by_topic.get(topic_id),
                "user_groups_preview": user_groups_by_topic.get(topic_id),
                "comment_preview": comment_preview.get(topic_id, []),
            }
        )

    return {
        "app_meta": {
            "title": "Reddit Topic Analysis Inspector",
            "subreddit": "r/Parenting",
            "analysis_scope": "Project Part 1 tasks 1.1-1.4",
            "stance_mode": "preview_sample",
        },
        "aggregate_stats": aggregate,
        "topic_run_metadata": topic_run,
        "stance_preview_metadata": stance_meta,
        "overview": {
            "topic_count": len(topics),
            "persistent_topics": persistent_count,
            "trending_topics": trending_count,
            "episodic_topics": episodic_count,
            "stance_preview_topics": len(stance_preview),
            "stance_preview_comments": stance_meta["comment_count_analyzed"],
        },
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
