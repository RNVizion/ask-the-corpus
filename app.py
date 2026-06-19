import time
from collections import defaultdict, deque

import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ---- config (the guardrail knobs) ----
MODEL = "claude-haiku-4-5"      # cheap + fast; the whole cost story
MAX_INPUT = 500                 # reject longer questions (bounds cost per call)
TOP_K = 5                       # chunks retrieved per question
MAX_TOKENS = 400                # caps answer length, so each call's cost is bounded
RATE = {"per_min": 6, "per_day": 60}

SYSTEM = (
    "You answer questions about the personal blog of Christian 'RNVizion' Smith. "
    "Use ONLY the context excerpts provided. If they don't contain the answer, say plainly "
    "that the blog doesn't cover that yet — never use outside knowledge or guess. "
    "Keep it concise, and name the post(s) your answer draws from."
)

SUGGESTED = [
    "What is squish?",
    "Why was a developer's job never really the code?",
    "What does constraint have to do with creativity?",
]

# ---- load the prebuilt index + embedder once ----
col = chromadb.PersistentClient(path="chroma").get_collection("corpus")
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # must match the ingest model
llm = Anthropic()   # reads ANTHROPIC_API_KEY from the environment

# ---- per-client rate limiter (in-memory) ----
_hits = defaultdict(deque)
def _rate_ok(key):
    now = time.time()
    dq = _hits[key]
    while dq and now - dq[0] > 86400:
        dq.popleft()
    if sum(1 for t in dq if now - t < 60) >= RATE["per_min"] or len(dq) >= RATE["per_day"]:
        return False
    dq.append(now)
    return True

def answer(question, request: gr.Request = None):
    question = (question or "").strip()
    if not question:
        return "Ask me something about the blog."
    if len(question) > MAX_INPUT:
        return f"Please keep your question under {MAX_INPUT} characters."
    key = request.client.host if request and request.client else "local"
    if not _rate_ok(key):
        return "You've hit the demo's rate limit for now — give it a minute, or try a suggested question."
    try:
        res = col.query(
            query_embeddings=embedder.encode([question]).tolist(),
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        docs, metas = res["documents"][0], res["metadatas"][0]
        if not docs:
            return "I don't have anything on that in the blog yet."
        context = "\n\n".join(f"[Post: {m.get('title', '?')}]\n{d}" for d, m in zip(docs, metas))
        resp = llm.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Context excerpts:\n\n{context}\n\nQuestion: {question}"}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        return f"DEBUG: {type(e).__name__}: {e}"

CSS = """
.gradio-container { background: #0a0a0f; }
h1, h2 { color: #d2bc93 !important; }
"""

with gr.Blocks(css=CSS, title="Ask the Corpus") as demo:
    gr.Markdown("# Ask the Corpus")
    gr.Markdown("Ask a question about Christian Smith's writing. Answers come only from the published posts — if it's not in the blog, it says so.")
    inp = gr.Textbox(label="Your question", placeholder="What is squish?", lines=2, max_lines=4)
    btn = gr.Button("Ask", variant="primary")
    out = gr.Markdown()
    gr.Examples(SUGGESTED, inputs=inp)
    btn.click(answer, inputs=inp, outputs=out)
    inp.submit(answer, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
