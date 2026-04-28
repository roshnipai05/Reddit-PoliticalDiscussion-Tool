"""Serve the local analysis app and expose lightweight JSON endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class AppRuntime:
    def __init__(self, chroma_dir: Path, model_cache_dir: Path) -> None:
        self.chroma_dir = chroma_dir
        self.model_cache_dir = model_cache_dir
        self._collections: tuple[Any, Any] | None = None

    def get_collections(self) -> tuple[Any, Any]:
        if self._collections is None:
            import rag_query

            self._collections = rag_query.get_collections(str(self.chroma_dir))
        return self._collections

    def bundle_path(self) -> Path:
        return APP_DIR / "data.bundle.json"

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        import rag_query

        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("Question is required.")

        lang = str(payload.get("lang") or "en")
        model = str(payload.get("model") or "both")
        query_type = str(payload.get("query_type") or "auto")
        post_collection, comment_collection = self.get_collections()
        return rag_query.run_rag_query(
            question=question,
            post_collection=post_collection,
            comment_collection=comment_collection,
            cache_dir=str(self.model_cache_dir),
            model=model,
            lang=lang,
            query_type=query_type,
        )

    def run_script(self, relative_script: str, args: list[str] | None = None) -> dict[str, Any]:
        command = [sys.executable, str(ROOT / relative_script), *(args or [])]
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def rebuild_bundle(self) -> dict[str, Any]:
        return self.run_script("scripts/build_app_bundle.py")

    def run_topic_analysis(self) -> dict[str, Any]:
        return self.run_script("scripts/topic_modeling_analysis.py")

    def run_stance_analysis(self) -> dict[str, Any]:
        return self.run_script(
            "scripts/topic_stance_analysis.py",
            ["--out-dir", "data/topic_stance_preview"],
        )

    def status(self) -> dict[str, Any]:
        bundle_exists = self.bundle_path().exists()
        topic_meta_path = DATA_DIR / "topic_analysis" / "run_metadata.json"
        stance_meta_path = DATA_DIR / "topic_stance_preview" / "run_metadata.json"
        index_meta_path = DATA_DIR / "chroma_db" / "index_metadata.json"

        def load_json(path: Path) -> Any:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        return {
            "app": {
                "bundle_exists": bundle_exists,
                "bundle_path": str(self.bundle_path()),
            },
            "rag": {
                "index_ready": index_meta_path.exists(),
                "index_metadata": load_json(index_meta_path),
            },
            "topic_analysis": {
                "ready": topic_meta_path.exists(),
                "run_metadata": load_json(topic_meta_path),
            },
            "stance_analysis": {
                "ready": stance_meta_path.exists(),
                "run_metadata": load_json(stance_meta_path),
            },
        }


class AppRequestHandler(SimpleHTTPRequestHandler):
    runtime: AppRuntime

    def __init__(self, *args, directory: str | None = None, runtime: AppRuntime | None = None, **kwargs):
        self.runtime = runtime or AppRuntime(DATA_DIR / "chroma_db", ROOT / "data" / "models" / "huggingface")
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/status":
            self._send_json(self.runtime.status())
            return
        if parsed.path == "/api/bundle":
            bundle_path = self.runtime.bundle_path()
            if not bundle_path.exists():
                self._send_json({"error": "Bundle not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(json.loads(bundle_path.read_text(encoding="utf-8")))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/query":
                self._send_json(self.runtime.query(payload))
                return
            if parsed.path == "/api/actions/rebuild-bundle":
                self._send_json(self.runtime.rebuild_bundle())
                return
            if parsed.path == "/api/actions/run-topic-analysis":
                self._send_json(self.runtime.run_topic_analysis())
                return
            if parsed.path == "/api/actions/run-stance-analysis":
                self._send_json(self.runtime.run_stance_analysis())
                return
            self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._send_json(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=HTTPStatus.BAD_REQUEST,
            )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--chroma-dir", default=str(DATA_DIR / "chroma_db"))
    parser.add_argument("--model-cache-dir", default=str(ROOT / "data" / "models" / "huggingface"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = AppRuntime(Path(args.chroma_dir), Path(args.model_cache_dir))
    handler = partial(AppRequestHandler, directory=str(APP_DIR), runtime=runtime)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving app on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
