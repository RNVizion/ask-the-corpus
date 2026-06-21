import re, requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

# PUBLISHED-ONLY sources — every entry is a live, public page on rnvizion.dev.
# Drafts and local files never go in here; what's published is what the bot knows.
# To add knowledge, publish the page first, then add its live URL below.
SOURCES = [
    {"id": "squish",                  "url": "https://rnvizion.dev/blog/squish/"},
    {"id": "i-lacked-the-tools",      "url": "https://rnvizion.dev/blog/i-lacked-the-tools/"},
    {"id": "the-job-was-never-coding","url": "https://rnvizion.dev/blog/the-job-was-never-coding/"},
    {"id": "the-margin",              "url": "https://rnvizion.dev/blog/the-margin/"},
    {"id": "bio",                     "url": "https://rnvizion.dev/bio/"},
    {"id": "resume",                   "url": "https://rnvizion.dev/resume/"},
    # add more published pages here as they go live.
]
CHUNK_WORDS, OVERLAP = 300, 50

def fetch_source(url):
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if not article:
        raise ValueError(f"no <article> at {url} — is it published?")
    bio = article.find("div", class_="bio")   # strip the author blurb on blog posts
    if bio: bio.decompose()
    t = soup.find("meta", property="og:title")
    title = t["content"] if t else url
    return title, re.sub(r"\s+", " ", article.get_text(" ", strip=True))

def chunk(text):
    words, step = text.split(), CHUNK_WORDS - OVERLAP
    for i in range(0, len(words), step):
        piece = words[i:i + CHUNK_WORDS]
        if piece: yield " ".join(piece)

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma")
    try: client.delete_collection("corpus")
    except Exception: pass
    col = client.get_or_create_collection("corpus")
    total = 0
    for src in SOURCES:
        sid, url = src["id"], src["url"]
        title, text = fetch_source(url)
        chunks = list(chunk(text))
        col.add(
            ids=[f"{sid}-{i}" for i in range(len(chunks))],
            embeddings=model.encode(chunks).tolist(),
            documents=chunks,
            metadatas=[{"source": sid, "title": title} for _ in chunks],
        )
        total += len(chunks)
        print(f"  {sid}: {len(chunks)} chunks ({title})")
    print(f"done — {total} chunks from {len(SOURCES)} sources → ./chroma")

if __name__ == "__main__":
    main()
