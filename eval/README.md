# Eval — does the corpus stay honest?

This suite measures the one claim the whole project makes: **answers come from the corpus, or not at all.** It runs every question in `cases.jsonl` through the real `app.answer()` and scores whether the system retrieves the right source, refuses what it doesn't cover, and never refuses what it does.

## What it scores

- **Retrieval accuracy** — for in-corpus questions, did the right source get pulled (matched by chunk-id prefix, e.g. `squish-3` → `squish`)?
- **Out-of-corpus refusal accuracy** — for questions the blog doesn't cover, did it return the exact denial line? This is the honesty thesis, measured.
- **False-refusal rate** — in-corpus questions it wrongly refused. Should be ~0.
- **Keyword groundedness** — a light proxy: did the answer contain an expected term? Directional only; it isn't an LLM judge.

The most interesting cases are the **traps**: questions about pending posts (`The Margin`, `The Job Was Never Coding`) and unpublished personal details. They're out-of-corpus *today*, so a correct system refuses them. When a pending post deploys and gets ingested, flip its case from `out_of_corpus` to `in_corpus`.

## Run it (from the repo root)

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # same key the Space uses
python eval/evaluate.py                   # writes eval/report.md, gates on thresholds
python eval/evaluate.py --report-only     # report without failing
python eval/evaluate.py --limit 10        # sample while iterating
pytest eval/test_eval.py -v               # the CI gate
```

It runs from the repo root because `app.py` opens `chroma/` by relative path. The per-client rate limiter is disabled for the eval only (it's infra, not answer quality); `app.py` is never modified.

## CI

`.github/workflows/eval.yml` runs the report and the pytest gate on any change to `app.py`, `ingest.py`, `sources.json`, `chroma/**`, or `eval/**`, and uploads `report.md` as an artifact. Add `ANTHROPIC_API_KEY` as a repository **Actions** secret (you already have it as a Space and Codespaces secret).

## Thresholds

The gate lives in `eval/thresholds.json`: retrieval ≥ 85%, out-of-corpus refusal ≥ 90%, false-refusal ≤ 10%. Edit that file to tune the bar; `evaluate.py` and the CI pytest both read it (and fall back to the same defaults if it's missing). Raise the bar as the corpus grows.
