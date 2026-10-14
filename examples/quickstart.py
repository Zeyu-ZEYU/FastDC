"""Quick start: offline stage and generation with the reference engine.

    python examples/quickstart.py --model Qwen/Qwen3-8B --r 0.02

1. calibrates the model (post-RoPE Q/K rotations from random tokens, whitened V basis),
2. builds an adaptive FastDC plan from one energy removal rate r,
3. generates the same prompt without compression and with FastDC and prints the KV cache sizes.
"""

import argparse

from fastdc.calibration import calibrate
from fastdc.engine import Engine
from fastdc.evaluate import encode, stop_ids
from fastdc.models import load_model, model_shape
from fastdc.plan import adaptive, full_plan

PROMPT = "Explain in three sentences why the KV cache of a language model grows with the context length."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--r", type=float, default=0.02, help="energy removal rate")
    ap.add_argument("--window", type=int, default=16384, help="context window covered by the calibration")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    calib = calibrate(model, tok, window=args.window, progress=True)
    plan = adaptive(calib, args.r)
    print("plan:", plan.summary())

    chat = getattr(tok, "chat_template", None) is not None
    ids = encode(tok, args.prompt, chat)
    for name, p in (("uncompressed", full_plan(model_shape(model.config))), ("FastDC", plan)):
        engine = Engine(model, p)
        out = engine.generate(ids, args.max_new_tokens, stop_ids(tok))
        print(f"\n=== {name}: KV cache {engine.cache_bytes() / 2**20:.2f} MiB "
              f"(uncompressed {engine.full_cache_bytes(engine.length) / 2**20:.2f} MiB)")
        print(tok.decode(out, skip_special_tokens=True))


if __name__ == "__main__":
    main()
