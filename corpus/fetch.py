"""Download PyTorch documentation HTML (stable docs + tutorials) into data/raw/.

Idempotent: a page already present on disk is not re-fetched, so an interrupted run
can simply be restarted. Writes data/manifest.json recording the docs version, the
fetch date, the full URL list and the page counts.

Usage:
    python -m corpus.fetch [--workers 8] [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

DOCS_HOST = "https://docs.pytorch.org"
STABLE_INDEX = f"{DOCS_HOST}/docs/stable/index.html"
DOCS_SITEMAP = f"{DOCS_HOST}/docs/stable/sitemap.xml"
TUTORIALS_SITEMAP = f"{DOCS_HOST}/tutorials/sitemap.xml"

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")
USER_AGENT = "rag-corpus-fetch/0.1 (+https://github.com/mzo111; polite crawler, cached locally)"

# Pages that are navigation/generated indexes rather than documentation content.
TUTORIAL_EXCLUDE = re.compile(
    r"/tutorials/(unstable/|unstable_index\.html|sg_execution_times\.html|genindex|search\.html|"
    r".*/sg_execution_times\.html)"
)
DOCS_EXCLUDE = re.compile(r"/(genindex|py-modindex|search|404)\.html$")


def resolve_stable_version(session: requests.Session) -> str:
    """Return the version directory that docs/stable/ currently redirects to (e.g. '2.14')."""
    resp = session.get(STABLE_INDEX, timeout=30)
    resp.raise_for_status()
    m = re.search(r'rel="canonical" href="\.\./([^/"]+)/index\.html"', resp.text)
    if m:
        return m.group(1)
    m = re.search(r"/docs/([^/]+)/index\.html", resp.url)
    if m and m.group(1) != "stable":
        return m.group(1)
    raise RuntimeError("could not determine the stable docs version from the redirect page")


def read_full_version(session: requests.Session, version_dir: str) -> str | None:
    """Best-effort full release string (e.g. '2.14.0') from the [source] links on an API page."""
    resp = session.get(f"{DOCS_HOST}/docs/{version_dir}/generated/torch.nn.Linear.html", timeout=30)
    if not resp.ok:
        return None
    m = re.search(rf"github\.com/pytorch/pytorch/blob/v({re.escape(version_dir)}\.\d+)/", resp.text)
    return m.group(1) if m else None


def parse_sitemap(xml: str) -> list[str]:
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def build_url_list(session: requests.Session, version_dir: str) -> tuple[list[str], list[str]]:
    docs_xml = session.get(DOCS_SITEMAP, timeout=60)
    docs_xml.raise_for_status()
    docs = []
    for u in parse_sitemap(docs_xml.text):
        u = u.replace("/docs/stable/", f"/docs/{version_dir}/")
        if not DOCS_EXCLUDE.search(u):
            docs.append(u)

    tut_xml = session.get(TUTORIALS_SITEMAP, timeout=60)
    tut_xml.raise_for_status()
    tutorials = [u for u in parse_sitemap(tut_xml.text) if not TUTORIAL_EXCLUDE.search(u)]
    return sorted(set(docs)), sorted(set(tutorials))


def cache_path(url: str, raw_dir: Path = RAW_DIR) -> Path:
    """Mirror the URL path under raw_dir: .../docs/2.14/generated/torch.nn.Linear.html"""
    path = urlparse(url).path.lstrip("/")
    if path.endswith("/") or not path:
        path += "index.html"
    return raw_dir / path


def fetch_one(
    session: requests.Session, url: str, dest: Path, force: bool = False, retries: int = 4
) -> tuple[str, str, int]:
    """Fetch url into dest -> (url, status, bytes); status is cached/fetched/failed."""
    if dest.exists() and not force:
        return url, "cached", dest.stat().st_size
    delay = 1.0
    last_err = ""
    for _ in range(retries):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                return url, "fetched", len(resp.content)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else delay)
                delay *= 2
                continue
            return url, f"failed: HTTP {resp.status_code}", 0
        except requests.RequestException as e:
            last_err = repr(e)
            time.sleep(delay)
            delay *= 2
    return url, f"failed: {last_err}", 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N pages (debug)")
    ap.add_argument("--force", action="store_true", help="re-download cached pages")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    version_dir = resolve_stable_version(session)
    full_version = read_full_version(session, version_dir)
    docs_urls, tut_urls = build_url_list(session, version_dir)
    print(
        f"stable -> {version_dir} ({full_version}); "
        f"{len(docs_urls)} docs, {len(tut_urls)} tutorials"
    )

    urls = docs_urls + tut_urls
    if args.limit:
        urls = urls[: args.limit]

    results: dict[str, tuple[str, int]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(fetch_one, session, u, cache_path(u, args.raw_dir), args.force)
            for u in urls
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            url, status, nbytes = fut.result()
            results[url] = (status, nbytes)
            if i % 200 == 0 or i == len(urls):
                counts = {}
                for s, _ in results.values():
                    counts[s.split(":")[0]] = counts.get(s.split(":")[0], 0) + 1
                print(f"[{i}/{len(urls)}] {counts} {time.time() - t0:.0f}s", flush=True)

    failures = sorted(u for u, (s, _) in results.items() if s.startswith("failed"))
    ok_docs = [u for u in docs_urls if u in results and not results[u][0].startswith("failed")]
    ok_tut = [u for u in tut_urls if u in results and not results[u][0].startswith("failed")]
    manifest = {
        "docs_version_dir": version_dir,
        "docs_version": full_version,
        "stable_url_prefix": f"{DOCS_HOST}/docs/stable/",
        "versioned_url_prefix": f"{DOCS_HOST}/docs/{version_dir}/",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {"docs": DOCS_SITEMAP, "tutorials": TUTORIALS_SITEMAP},
        "pages": {
            "docs": len(ok_docs),
            "tutorials": len(ok_tut),
            "total": len(ok_docs) + len(ok_tut),
        },
        "bytes": sum(n for _, n in results.values()),
        "failures": [{"url": u, "error": results[u][0]} for u in failures],
        "urls": ok_docs + ok_tut,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"done: {manifest['pages']} pages, {manifest['bytes'] / 1e6:.0f} MB, "
        f"{len(failures)} failures -> {args.manifest}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
