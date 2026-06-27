"""
test_eval.py — pytest gate over the eval metrics.

Runs the full eval once, then asserts the aggregate thresholds. This is the
CI gate; evaluate.py is the human-readable report. Needs ANTHROPIC_API_KEY.

    pytest eval/test_eval.py -v
"""
import pytest

from eval import evaluate


@pytest.fixture(scope="module")
def metrics():
    m, rows = evaluate.run()
    evaluate.write_report(m, rows)   # leave a report artifact behind too
    return m


def test_retrieval_accuracy(metrics):
    assert metrics["retrieval_accuracy"] is not None
    assert metrics["retrieval_accuracy"] >= evaluate.THRESHOLDS["retrieval_accuracy"], metrics


def test_out_of_corpus_refusal(metrics):
    # The honesty thesis, measured: out-of-corpus questions must get the denial line.
    assert metrics["ooc_refusal_accuracy"] >= evaluate.THRESHOLDS["ooc_refusal_accuracy"], metrics


def test_no_false_refusals(metrics):
    # In-corpus questions must NOT be refused.
    assert metrics["false_refusal_rate"] <= evaluate.THRESHOLDS["false_refusal_rate"], metrics
