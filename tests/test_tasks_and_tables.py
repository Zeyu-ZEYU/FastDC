import pytest

from fastdc.engine import KeyformerConfig, KVQuantConfig
from fastdc.fp_table import overall_rate, select
from fastdc.tasks import Example, Task, edit_similarity, rouge2


def test_split_is_disjoint_and_20_80():
    task = Task("t", "accuracy", [Example([str(i)], [i], 1) for i in range(50)], lambda o, r: 0.0)
    cal, ev = task.split("calibration"), task.split("evaluation")
    assert len(cal) == 10 and len(ev) == 40
    assert {e.references[0] for e in cal}.isdisjoint({e.references[0] for e in ev})
    assert [e.references[0] for e in task.split("calibration")] == [e.references[0] for e in cal]
    assert len(task.split("all")) == 50


def test_metrics():
    assert edit_similarity("abc", "abc") == 1.0
    assert edit_similarity("abc", "abd") == pytest.approx(2 / 3)
    assert edit_similarity("", "") == 1.0
    assert rouge2("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert rouge2("completely different words here", "the cat sat on the mat") == 0.0


def test_overall_rate_and_select():
    assert overall_rate(0.5, None, None) == pytest.approx(0.5)
    assert overall_rate(0.5, KeyformerConfig(keep_ratio=0.5), None) == pytest.approx(0.75)
    assert overall_rate(0.4, None, KVQuantConfig(bits=4)) == pytest.approx(1 - 0.6 * 0.25)
    table = {"rows": [dict(x=0.01, kv_rate=0.2, relative=1.0), dict(x=0.02, kv_rate=0.3, relative=0.995),
                      dict(x=0.04, kv_rate=0.4, relative=0.97)]}
    assert select(table, 0.99)["x"] == 0.02
    assert select(table, 0.95)["x"] == 0.04
    assert select(table, 1.01) is None
