"""Generate with vLLM on the FastDC attention backend (offline inference).

    python examples/vllm_generate.py --plan plan.safetensors --model Qwen/Qwen3-0.6B

The plan comes from the offline stage, for example
    fastdc calibrate --model Qwen/Qwen3-0.6B --out calib.safetensors
    fastdc plan --calibration calib.safetensors --method fastdc --r 0.01 --out plan.safetensors
A plan of method "full" serves the model uncompressed through the same backend. The equivalent server is
    vllm serve Qwen/Qwen3-0.6B --attention-backend CUSTOM --additional-config '{"fastdc": {"plan": "plan.safetensors"}}'
"""

from __future__ import annotations

import argparse

from fastdc.plan import CompressionPlan

PROMPTS = [
    "The capital of France is",
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about the sea.\n",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, help="CompressionPlan file (.safetensors)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", action="append", help="prompt (repeatable); a few defaults otherwise")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--attention", choices=("kernels", "reference"), default="kernels")
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                        help="load-balanced kernel schedules")
    parser.add_argument("--max-model-len", type=int, default=None, help="default: min(4096, the model's context)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true", help="no torch.compile and no CUDA graphs")
    parser.add_argument("--raw", action="store_true", help="send the prompts without the model's chat template")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    plan = CompressionPlan.load(args.plan)
    print(f"plan: {plan.summary()}")
    max_model_len = args.max_model_len or min(4096, plan.shape.max_positions)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        attention_backend="CUSTOM",
        additional_config={"fastdc": {"plan": args.plan, "attention": args.attention, "balance": args.balance}},
    )
    prompts = args.prompt or PROMPTS
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    if llm.get_tokenizer().chat_template is not None and not args.raw:
        outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sampling,
                           chat_template_kwargs={"enable_thinking": False})
    else:
        outputs = llm.generate(prompts, sampling)
    for prompt, output in zip(prompts, outputs):
        print(f"\n>>> {prompt!r}\n{output.outputs[0].text}")
    cache = llm.llm_engine.vllm_config.cache_config
    print(f"\nKV cache: {cache.num_gpu_blocks} blocks of {cache.block_size} tokens "
          f"({cache.num_gpu_blocks * cache.block_size} tokens)")


if __name__ == "__main__":
    main()
