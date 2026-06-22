import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

HERE = Path(__file__).resolve().parent
SOURCES_FILE = HERE / "sources.json"
CHUNK_WORDS, OVERLAP = 300, 50


class SkipSource(Exception):
    """Raised when a source can't be ingested (dead URL, not published, etc.)."""


def load_sources():
    """Read the live source list from sources.json (the '_pending' list is ignored)."""
    data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    return data.get("sources", [])


def fetch_source(url):
    """Fetch and clean a source page. Returns (title, text).
    Raises SkipSource if the page is missing, errored, or has no readable content."""
    try:
        resp = requests.get(url, timeout=20)
    except requests.RequestException as e:
        raise SkipSource(f"request failed: {e}")
    if resp.status_code != 200:
        raise SkipSource(f"HTTP {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")

    # Prefer <article> (blog posts); fall back to <main>, then <body>, so non-post
    # pages (résumé, bio, future pages) are ingestible without bending their HTML.
    container = soup.find("article") or soup.find("main") or soup.body
    if container is None:
        raise SkipSource("no readable content")
    # When we fall through to <main>/<body>, drop site chrome so the menu and
    # boilerplate don't pollute the chunks. (An <article> rarely holds these, so
    # this is a safe no-op for posts. <header> is left alone — it can carry a
    # page's title/intro, and the nav already lives in <nav>.)
    for tag in container.select("nav, footer, script, style"):
        tag.decompose()
    bio = container.find("div", class_="bio")  # author blurb on blog posts
    if bio:
        bio.decompose()

    t = soup.find("meta", property="og:title")
    title = t["content"] if t else url
    return title, re.sub(r"\s+", " ", container.get_text(" ", strip=True))


def chunk(text):
    words, step = text.split(), CHUNK_WORDS - OVERLAP
    for i in range(0, len(words), step):
        piece = words[i:i + CHUNK_WORDS]
        if piece:
            yield " ".join(piece)


def main():
    sources = load_sources()
    if not sources:
        sys.exit(f"no sources found in {SOURCES_FILE}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma")
    try:
        client.delete_collection("corpus")
    except Exception:
        pass
    col = client.get_or_create_collection("corpus")

    total = 0
    ingested, skipped = [], []
    for src in sources:
        sid, url = src["id"], src["url"]
        try:
            title, text = fetch_source(url)
        except SkipSource as why:
            skipped.append((sid, str(why)))
            print(f"  SKIP {sid}: {why}  ({url})")
            continue
        chunks = list(chunk(text))
        col.add(
            ids=[f"{sid}-{i}" for i in range(len(chunks))],
            embeddings=model.encode(chunks).tolist(),
            documents=chunks,
            metadatas=[{"source": sid, "title": title} for _ in chunks],
        )
        total += len(chunks)
        ingested.append(sid)
        print(f"  ok   {sid}: {len(chunks)} chunks ({title})")

    print(f"\ndone — {total} chunks from {len(ingested)} sources → ./chroma")
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(f"{s} ({w})" for s, w in skipped))
        print("the corpus excludes these for now; deploy or fix them, then re-run.")


if __name__ == "__main__":
    main()
