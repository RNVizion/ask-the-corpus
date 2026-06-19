import re, requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

# PUBLISHED-ONLY allowlist — put your real live slugs here:
SLUGS = [
    "squish",
    "i-lacked-the-tools",
    # "the-job-was-never-coding",
    # "the-margin",
    # "without-a-compass",
]
BASE = "https://rnvizion.dev/blog/{}/"
CHUNK_WORDS, OVERLAP = 300, 50

def fetch_article(slug):
    html = requests.get(BASE.format(slug), timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if not article:
        raise ValueError(f"no <article> for {slug} — is it published?")
    bio = article.find("div", class_="bio")
    if bio: bio.decompose()
    t = soup.find("meta", property="og:title")
    title = t["content"] if t else slug
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
    for slug in SLUGS:
        title, text = fetch_article(slug)
        chunks = list(chunk(text))
        col.add(
            ids=[f"{slug}-{i}" for i in range(len(chunks))],
            embeddings=model.encode(chunks).tolist(),
            documents=chunks,
            metadatas=[{"slug": slug, "title": title} for _ in chunks],
        )
        total += len(chunks)
        print(f"  {slug}: {len(chunks)} chunks ({title})")
    print(f"done — {total} chunks from {len(SLUGS)} posts → ./chroma")

if __name__ == "__main__":
    main()
