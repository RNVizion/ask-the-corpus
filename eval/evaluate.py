"""
evaluate.py — honesty + retrieval eval for Ask the Corpus.

Runs every case in eval/cases.jsonl through the REAL pipeline in app.py and scores:

  - retrieval accuracy : for in-corpus questions, did the right source get retrieved?
  - refusal accuracy   : out-of-corpus -> did it return the exact denial line?
                         in-corpus     -> did it NOT refuse (no false refusals)?
  - keyword groundedness: a light proxy — did the answer contain an expected term?
                         (cheap signal, not an LLM judge; treat as directional)

Run from the repo root (app.py uses a relative chroma/ path):

    python eval/evaluate.py                # gate on thresholds, exit 1 on fail
    python eval/evaluate.py --report-only  # never fail; just write the report
    python eval/evaluate.py --limit 10     # sample the first N cases

Needs ANTHROPIC_API_KEY in the environment (same key the Space uses).
Writes eval/report.md and eval/results.json.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
CASES = EVAL_DIR / "cases.jsonl"
REPORT_MD = EVAL_DIR / "report.md"
RESULTS_JSON = EVAL_DIR / "results.json"

# app.py loads the chroma index from a relative "chroma/" path and reads the
# question's client host for rate limiting; run everything as if from the repo root.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import app  # noqa: E402  (the real pipeline: answer(), col, embedder, TOP_K, SYSTEM)

# The eval hammers answer() in a tight loop; the per-client rate limiter would
# trip after a handful of calls and poison the run. It's infra, not answer
# quality, so we disable it for the eval only. app.py is left untouched.
app._rate_ok = lambda key=None: True

DENIAL = "The corpus has knowledge, but the information you seek will not be found here."

# Valid source ids (the prefix on every chunk id, e.g. "squish-3" -> "squish").
LIVE_SOURCE_IDS = {
    s["id"] for s in json.loads((REPO_ROOT / "sources.json").read_text(encoding="utf-8")).get("sources", [])
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def is_refusal(answer_text: str) -> bool:
    return _norm(DENIAL) in _norm(answer_text)


def retrieved_source_ids(question: str):
    """Re-run retrieval the same way app.answer does, and map chunk ids -> source ids.
    Chunk ids look like '{source_id}-{n}', so rsplit on the last hyphen recovers
    the source id even when the id itself contains hyphens (i-lacked-the-tools-3)."""
    res = app.col.query(
        query_embeddings=app.embedder.encode([question]).tolist(),
        n_results=app.TOP_K,
        include=["metadatas"],
    )
    ids = res.get("ids", [[]])[0]
    out = []
    for cid in ids:
        sid = cid.rsplit("-", 1)[0]
        out.append(sid if sid in LIVE_SOURCE_IDS else cid)
    return out


def load_cases(limit=None):
    rows = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def expected_ids(case):
    exp = case.get("expect_source")
    if exp is None:
        return []
    return exp if isinstance(exp, list) else [exp]


def run(limit=None):
    cases = load_cases(limit)
    rows, calls = [], 0

    for c in cases:
        kind = c["kind"]
        q = c["question"]
        retrieved = retrieved_source_ids(q)
        ans = app.answer(q)          # the real path, denial line included
        calls += 1
        refused = is_refusal(ans)

        row = {
            "id": c["id"], "kind": kind, "question": q,
            "retrieved": retrieved, "refused": refused,
            "answer_preview": _norm(ans)[:120],
        }

        if kind == "in_corpus":
            exp = expected_ids(c)
            row["expect_source"] = exp
            row["retrieval_hit"] = any(e in retrieved for e in exp)
            row["false_refusal"] = refused          # in-corpus should NEVER refuse
            kws = [k.lower() for k in c.get("keywords", [])]
            row["keyword_hit"] = (any(k in _norm(ans) for k in kws) if kws else None)
            row["pass"] = row["retrieval_hit"] and not refused
        else:  # out_of_corpus
            row["refusal_correct"] = refused        # should refuse
            row["pass"] = refused
            if c.get("note"):
                row["note"] = c["note"]

        rows.append(row)

    in_rows = [r for r in rows if r["kind"] == "in_corpus"]
    out_rows = [r for r in rows if r["kind"] == "out_of_corpus"]

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    kw_rows = [r for r in in_rows if r.get("keyword_hit") is not None]
    metrics = {
        "total_cases": len(rows),
        "claude_calls": calls,
        "retrieval_accuracy": pct([r["retrieval_hit"] for r in in_rows]),
        "false_refusal_rate": pct([r["false_refusal"] for r in in_rows]),
        "ooc_refusal_accuracy": pct([r["refusal_correct"] for r in out_rows]),
        "keyword_groundedness": pct([r["keyword_hit"] for r in kw_rows]),
        "overall_pass_rate": pct([r["pass"] for r in rows]),
    }
    return metrics, rows


def write_report(metrics, rows):
    RESULTS_JSON.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")

    def b(v):
        return "✅" if v else "❌"

    lines = ["# Ask the Corpus — Eval Report", ""]
    lines += [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Retrieval accuracy (in-corpus) | {metrics['retrieval_accuracy']}% |",
        f"| Out-of-corpus refusal accuracy | {metrics['ooc_refusal_accuracy']}% |",
        f"| False-refusal rate (in-corpus) | {metrics['false_refusal_rate']}% |",
        f"| Keyword groundedness (proxy) | {metrics['keyword_groundedness']}% |",
        f"| Overall pass rate | {metrics['overall_pass_rate']}% |",
        f"| Cases / Claude calls | {metrics['total_cases']} / {metrics['claude_calls']} |",
        "",
        "## In-corpus",
        "| id | retrieved right source | refused? | keyword | pass |",
        "| --- | :---: | :---: | :---: | :---: |",
    ]
    for r in [r for r in rows if r["kind"] == "in_corpus"]:
        kw = "—" if r.get("keyword_hit") is None else b(r["keyword_hit"])
        lines.append(f"| {r['id']} | {b(r['retrieval_hit'])} | {'⚠️' if r['refused'] else '—'} | {kw} | {b(r['pass'])} |")

    lines += ["", "## Out-of-corpus (should refuse)", "| id | refused? | pass |", "| --- | :---: | :---: |"]
    for r in [r for r in rows if r["kind"] == "out_of_corpus"]:
        lines.append(f"| {r['id']} | {b(r['refused'])} | {b(r['pass'])} |")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Gate thresholds — starting points; tune as the corpus grows.
THRESHOLDS = {
    "retrieval_accuracy": 85.0,     # >= this
    "ooc_refusal_accuracy": 90.0,   # >= this
    "false_refusal_rate": 10.0,     # <= this
}


def gate(metrics):
    failures = []
    if (metrics["retrieval_accuracy"] or 0) < THRESHOLDS["retrieval_accuracy"]:
        failures.append(f"retrieval_accuracy {metrics['retrieval_accuracy']}% < {THRESHOLDS['retrieval_accuracy']}%")
    if (metrics["ooc_refusal_accuracy"] or 0) < THRESHOLDS["ooc_refusal_accuracy"]:
        failures.append(f"ooc_refusal_accuracy {metrics['ooc_refusal_accuracy']}% < {THRESHOLDS['ooc_refusal_accuracy']}%")
    if (metrics["false_refusal_rate"] or 0) > THRESHOLDS["false_refusal_rate"]:
        failures.append(f"false_refusal_rate {metrics['false_refusal_rate']}% > {THRESHOLDS['false_refusal_rate']}%")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="write the report but never exit non-zero")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    args = ap.parse_args()

    metrics, rows = run(limit=args.limit)
    write_report(metrics, rows)

    print(json.dumps(metrics, indent=2))
    print(f"\nReport: {REPORT_MD}")

    failures = gate(metrics)
    if failures and not args.report_only:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nGate passed." if not failures else "\n(report-only: gate not enforced)")


if __name__ == "__main__":
    main()
