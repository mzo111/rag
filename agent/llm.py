"""Ollama HTTP client with an on-disk response cache.

Talks to Ollama's ``/api/generate`` with :mod:`requests` and nothing else - no LangChain, no
LangGraph, no client library. The surface needed here is one POST.

**The cache is the reproducibility artifact.** Every response is written to a JSON file keyed
by ``sha256(model, prompt, query)`` and that file is committed, so the evaluation replays
without Ollama installed and without a GPU. Because the key covers the full prompt text,
editing a prompt in :mod:`agent.prompts` invalidates its entries rather than silently reusing
stale generations - which matters when the prompt is supposed to be frozen.

Determinism: temperature is 0 and fixed. That makes a *fresh* run reproducible up to Ollama's
own numerics; a *cached* run is exactly reproducible, which is the stronger property and the
one the committed cache provides.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_CACHE = Path(__file__).parent / "cache" / "ollama.json"
# Fixed a priori: greedy decoding, so the only nondeterminism left is Ollama's own numerics.
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120


class OllamaError(RuntimeError):
    """Ollama is unreachable, the model is missing, or a call failed."""


@dataclass
class Response:
    """One generation, whether it came from Ollama or from the cache."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ResponseCache:
    """JSON-file cache of model responses. Small enough to commit and read whole."""

    def __init__(self, path: Path = DEFAULT_CACHE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt cache must not take the run down; it is an optimisation.
                self._data = {}

    @staticmethod
    def key(model: str, prompt: str, query: str) -> str:
        h = hashlib.sha256()
        for part in (model, prompt, query):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Response | None:
        row = self._data.get(key)
        if row is None:
            return None
        return Response(
            text=row["text"],
            prompt_tokens=row.get("prompt_tokens", 0),
            completion_tokens=row.get("completion_tokens", 0),
            seconds=row.get("seconds", 0.0),
            cached=True,
        )

    def put(self, key: str, response: Response, *, meta: dict | None = None) -> None:
        with self._lock:
            self._data[key] = {
                "text": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "seconds": round(response.seconds, 4),
                **(meta or {}),
            }

    def save(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.path.write_text(
                json.dumps(self._data, indent=1, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            return len(self._data)

    def __len__(self) -> int:
        return len(self._data)


class OllamaClient:
    """Minimal Ollama client: cache lookup, then one POST to /api/generate."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache: ResponseCache | None = None,
        temperature: float = TEMPERATURE,
        timeout: int = REQUEST_TIMEOUT,
        offline: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache = cache if cache is not None else ResponseCache()
        self.temperature = temperature
        self.timeout = timeout
        # offline=True refuses to call Ollama at all: a cache miss becomes an error rather
        # than a silent live call, which is what makes a committed-cache replay verifiable.
        self.offline = offline
        self.calls = 0
        self.cache_hits = 0

    def check_available(self) -> None:
        """Raise :class:`OllamaError` with a fixable message if we cannot serve live calls."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url}: {exc}\n"
                "Start it with `ollama serve`, or run against the committed cache."
            ) from exc
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model not in names:
            raise OllamaError(
                f"model {self.model!r} is not pulled on {self.base_url}.\n"
                f"Pull it with `ollama pull {self.model}`.\n"
                f"Available: {', '.join(sorted(names)) or '(none)'}"
            )

    def generate(self, prompt: str, *, query: str, json_mode: bool = True) -> Response:
        """Return a completion, from cache when possible."""
        key = self.cache.key(self.model, prompt, query)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        if self.offline:
            raise OllamaError(
                f"cache miss for query {query!r} with offline=True. The committed cache does "
                "not cover this call; re-run with a live Ollama to extend it."
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"
        started = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Ollama call failed against {self.base_url}: {exc}\n"
                "Check `ollama serve` is running and the model is pulled."
            ) from exc
        elapsed = time.perf_counter() - started
        self.calls += 1

        response = Response(
            text=body.get("response", ""),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            seconds=elapsed,
            cached=False,
        )
        self.cache.put(key, response, meta={"model": self.model, "query": query})
        return response
