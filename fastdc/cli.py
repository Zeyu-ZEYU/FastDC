"""Command-line interface of the offline stage and of the reference engine.

    fastdc calibrate  --model M --out calib.safetensors            steps 1-2: R and singular values
    fastdc plan       --calibration C --method fastdc --r 0.02 --out plan.safetensors
    fastdc sweep      --model M --calibration C --task gsm8k --out table.json     F_p table (step 3)
    fastdc select     --table table.json --threshold 0.99 --calibration C --out plan.safetensors
    fastdc evaluate   --model M --plan plan.safetensors --task gsm8k
    fastdc generate   --model M --plan plan.safetensors --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def _model(args):
    from fastdc.models import load_model

    return load_model(args.model, device=args.device)


def _task(args, tokenizer):
    from fastdc.tasks import load_task

    kwargs = json.loads(args.task_args) if args.task_args else {}
    return load_task(args.task, tokenizer=tokenizer, **kwargs)


def cmd_calibrate(args):
    from fastdc.calibration import calibrate

    model, tok = _model(args)
    texts = None
    if args.calibration_text != "random":
        texts = _task(argparse.Namespace(task=args.calibration_text, task_args=args.task_args), tok).calibration_texts
    calib = calibrate(model, tok, num_positions=args.positions, seq_len=args.seq_len, texts=texts, seed=args.seed,
                      window=args.window, progress=True, whiten=not args.no_whiten, whiten_tokens=args.whiten_tokens)
    calib.save(args.out)
    print(json.dumps(calib.meta))


def _build_plan(calib, method, r=None, p=None):
    from fastdc.plan import adaptive, full_plan, uniform

    if method == "full":
        return full_plan(calib.shape)
    if (r is None) == (p is None):
        raise SystemExit("give exactly one of --r (adaptive) or --p (uniform)")
    return adaptive(calib, r, method) if r is not None else uniform(calib, p, method)


def cmd_plan(args):
    from fastdc.calibration import Calibration

    plan = _build_plan(Calibration.load(args.calibration), args.method, args.r, args.p)
    plan.save(args.out)
    print(json.dumps(plan.summary()))


def cmd_sweep(args):
    from fastdc.calibration import Calibration
    from fastdc.fp_table import sweep

    model, tok = _model(args)
    grid = [float(x) for x in args.grid.split(",")] if args.grid else None
    table = sweep(model, tok, Calibration.load(args.calibration), _task(args, tok), args.method, args.mode, grid,
                  args.kvquant, args.keyformer, args.split, args.limit, args.chat, progress=True)
    _write_json(args.out, table)
    print(f"wrote {args.out}")


def cmd_select(args):
    from fastdc.calibration import Calibration
    from fastdc.fp_table import select

    with open(args.table) as f:
        table = json.load(f)
    row = select(table, args.threshold)
    if row is None:
        raise SystemExit(f"no entry of {args.table} reaches {args.threshold:.2%} of the uncompressed accuracy")
    calib = Calibration.load(args.calibration)
    plan = _build_plan(calib, table["method"], row["x"] if table["mode"] == "adaptive" else None,
                       row["x"] if table["mode"] == "uniform" else None)
    plan.info.update(task=table["task"], threshold=args.threshold, accuracy=row["accuracy"], relative=row["relative"])
    plan.save(args.out)
    print(json.dumps(dict(selected=row, plan=plan.summary())))


def _engine(args, model, tok):
    from fastdc.engine import Engine, KeyformerConfig, KVQuantConfig
    from fastdc.models import model_shape
    from fastdc.plan import CompressionPlan, full_plan

    plan = CompressionPlan.load(args.plan) if args.plan else full_plan(model_shape(model.config))
    kf = KeyformerConfig(keep_ratio=args.keyformer) if args.keyformer else None
    kq = KVQuantConfig() if args.kvquant else None
    return Engine(model, plan, kvquant=kq, keyformer=kf, tokenizer=tok)


def cmd_evaluate(args):
    from fastdc.evaluate import evaluate

    model, tok = _model(args)
    res = evaluate(_engine(args, model, tok), tok, _task(args, tok), args.split, args.limit, args.chat,
                   progress=True)
    out = res.to_dict()
    out.pop("per_example")
    print(json.dumps(out))
    if args.out:
        _write_json(args.out, res.to_dict())


def cmd_generate(args):
    from fastdc.evaluate import encode, stop_ids

    model, tok = _model(args)
    engine = _engine(args, model, tok)
    chat = getattr(tok, "chat_template", None) is not None and not args.raw
    out = engine.generate(encode(tok, args.prompt, chat), args.max_new_tokens, stop_ids(tok))
    print(tok.decode(out, skip_special_tokens=True))
    print(f"[KV cache: {engine.cache_bytes() / 2**20:.2f} MiB stored, "
          f"{engine.full_cache_bytes(engine.length) / 2**20:.2f} MiB uncompressed]", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fastdc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, model=True):
        if model:
            p.add_argument("--model", required=True, help="Hugging Face model id or path")
            p.add_argument("--device", default="cuda")
        p.add_argument("--task-args", default=None, help="JSON keyword arguments of the task loader")

    p = sub.add_parser("calibrate", help="offline steps 1-2")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=16384, help="sampled post-RoPE positions")
    p.add_argument("--seq-len", type=int, default=8192, help="length of one calibration sequence")
    p.add_argument("--window", type=int, default=None, help="context window to tile (default: the model's)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-text", default="random", help="'random' or a task name whose prompts are used")
    p.add_argument("--no-whiten", action="store_true", help="plain SVD of W_V / W_K instead of the whitened one")
    p.add_argument("--whiten-tokens", type=int, default=65536)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("plan", help="build a compression plan")
    p.add_argument("--calibration", required=True)
    p.add_argument("--method", choices=("fastdc", "palu", "full"), default="fastdc")
    p.add_argument("--r", type=float, default=None, help="energy removal rate (adaptive rates)")
    p.add_argument("--p", type=float, default=None, help="uniform compression rate")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)

    def run_opts(p):
        p.add_argument("--task", required=True)
        p.add_argument("--split", default="evaluation", choices=("calibration", "evaluation", "all"))
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--chat", default="auto", choices=("auto", "never"))

    p = sub.add_parser("sweep", help="build the F_p table of one task")
    common(p)
    run_opts(p)
    p.set_defaults(split="calibration")
    p.add_argument("--calibration", required=True)
    p.add_argument("--method", choices=("fastdc", "palu"), default="fastdc")
    p.add_argument("--mode", choices=("adaptive", "uniform"), default="adaptive")
    p.add_argument("--grid", default=None, help="comma-separated r (adaptive) or p (uniform) values")
    p.add_argument("--kvquant", action="store_true")
    p.add_argument("--keyformer", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("select", help="plan for an accuracy threshold from an F_p table")
    p.add_argument("--table", required=True)
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--calibration", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_select)

    for name, func in (("evaluate", cmd_evaluate), ("generate", cmd_generate)):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--plan", default=None, help="compression plan (default: uncompressed)")
        p.add_argument("--kvquant", action="store_true", help="4-bit quantization of the stored K and V")
        p.add_argument("--keyformer", type=float, default=None, help="token eviction keeping this share of tokens")
        if name == "evaluate":
            run_opts(p)
            p.add_argument("--out", default=None)
        else:
            p.add_argument("--prompt", required=True)
            p.add_argument("--max-new-tokens", type=int, default=128)
            p.add_argument("--raw", action="store_true", help="do not apply the chat template")
        p.set_defaults(func=func)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
