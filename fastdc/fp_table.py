"""The F_p table: accuracy constraints -> per-head compression rates (offline stage).

For a grid of energy removal rates r (adaptive methods) or uniform rates p (uniform methods), the sweep
builds each plan, measures the accuracy on the task's calibration subset, and records
(r or p, overall KV compression rate, accuracy, accuracy relative to the uncompressed model).
`select` returns the most aggressive entry that still meets an accuracy threshold (e.g. 0.99 = 99% of the
uncompressed accuracy); at serving time this is a table lookup.

Combined methods: with Keyformer-style eviction (KF-*), eviction and dimensionality compression contribute
equally to the overall rate (the eviction keeps 1 - p of the tokens); with KVQuant-style quantization (KVQ-*),
the compressed K and V are stored in 4 bits, so the overall rate is 1 - (1 - p) * 4 / 16.
"""

from __future__ import annotations

import json

from fastdc.calibration import Calibration
from fastdc.engine import Engine, KeyformerConfig, KVQuantConfig
from fastdc.evaluate import evaluate
from fastdc.plan import adaptive, full_plan, uniform

DEFAULT_R_GRID = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1)
DEFAULT_P_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def overall_rate(dc_rate: float, keyformer: KeyformerConfig | None, kvquant: KVQuantConfig | None) -> float:
    keep = 1.0 - dc_rate
    if keyformer is not None:
        keep *= keyformer.keep_ratio
    if kvquant is not None:
        keep *= kvquant.bits / 16
    return 1.0 - keep


def sweep(model, tokenizer, calib: Calibration, task, method: str = "fastdc", mode: str = "adaptive",
          grid=None, kvquant: bool = False, keyformer: bool = False, split: str = "calibration",
          limit: int | None = None, chat: str = "auto", progress: bool = False) -> dict:
    grid = list(grid or (DEFAULT_R_GRID if mode == "adaptive" else DEFAULT_P_GRID))
    base = evaluate(Engine(model, full_plan(calib.shape)), tokenizer, task, split, limit, chat, progress=progress)
    rows = []
    for x in grid:
        plan = adaptive(calib, x, method) if mode == "adaptive" else uniform(calib, x, method)
        kf = KeyformerConfig(keep_ratio=1.0 - plan.kv_rate) if keyformer else None
        kq = KVQuantConfig() if kvquant else None
        engine = Engine(model, plan, kvquant=kq, keyformer=kf, tokenizer=tokenizer)
        res = evaluate(engine, tokenizer, task, split, limit, chat, progress=progress)
        rows.append(dict(x=x, dc_rate=plan.kv_rate, kv_rate=overall_rate(plan.kv_rate, kf, kq),
                         measured_kv_rate=res.measured_kv_rate, accuracy=res.score,
                         relative=res.score / base.score if base.score else 0.0))
        if progress:
            print(json.dumps(rows[-1]), flush=True)
    return dict(task=task.name, metric=task.metric, method=method, mode=mode, kvquant=kvquant, keyformer=keyformer,
                split=split, num_examples=base.num_examples, baseline=base.score, rows=rows)


def select(table: dict, threshold: float) -> dict | None:
    """Entry with the highest overall KV compression rate whose relative accuracy is >= threshold."""
    ok = [row for row in table["rows"] if row["relative"] >= threshold]
    return max(ok, key=lambda row: row["kv_rate"]) if ok else None
