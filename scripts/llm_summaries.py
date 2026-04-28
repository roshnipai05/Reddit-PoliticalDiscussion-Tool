from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from dotenv import load_dotenv


load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SLEEP_SECONDS = 3.0
_CALL_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0


def _clean_api_key(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    return raw_value.strip().strip('"').strip("'").strip()


def groq_available() -> bool:
    return bool(_clean_api_key(os.environ.get("GROQ_API_KEY", "")))


def groq_json_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 700,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> dict[str, Any]:
    from groq import Groq  # type: ignore

    api_key = _clean_api_key(os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    global _LAST_CALL_TS
    with _CALL_LOCK:
        elapsed = time.time() - _LAST_CALL_TS
        if elapsed < sleep_seconds:
            time.sleep(sleep_seconds - elapsed)
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        _LAST_CALL_TS = time.time()

    content = response.choices[0].message.content.strip()
    return json.loads(content)
