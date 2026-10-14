# FastDC

**FastDC: Fast KV Dimensionality Compression for Decompression-Free and Attention-Accelerated LLM Inference**
(ACM Symposium on Cloud Computing, SoCC '26)

FastDC compresses the Key-Value (KV) cache of large language models with Rotary Positional Embedding (RoPE)
along the head dimension. It compresses the query (Q) and key (K) *after* RoPE with a rotation matrix R
that is common to Q and K, specific to the model, and computed offline from random tokens. Attention then
runs directly on the compressed Q, K and V, so there is no decompression step and attention itself gets
cheaper. Compression rates adapt per head and layer under an accuracy constraint, and the attention kernels
balance the uneven work that heads of different widths produce.

This repository contains the offline stage (calibration, rate selection, accuracy constraints), a
reference engine on Hugging Face models, Triton attention kernels for the compressed paged cache, a vLLM
integration, the compared methods, and the scripts of the experiments.

## How FastDC works, and where the code is

| Step | What happens | Code |
|---|---|---|
| Offline 1-2 | Feed random tokens, capture post-RoPE Q and K per layer, stack the queries of a GQA group with copies of its key head, and decompose them with SVD: singular values and the rotation R of every KV head | `fastdc/calibration.py` |
| Offline 3 | One matrix energy removal rate r gives every head its own compression rate (Eq. 1); V uses the whitened decomposition of W_V | `fastdc/rates.py`, `fastdc/plan.py` |
| Offline 3 | The F_p table maps an accuracy constraint to the per-head rates (sweep on a 20% calibration split) | `fastdc/fp_table.py` |
| Online 4-6 | Q and K are rotated by R and truncated after RoPE, V is produced compressed (V' = E A_V), and the V basis is folded into the output projection | `fastdc/vllm_plugin/`, `fastdc/engine.py` |
| Online 7 | Attention on the compressed paged cache with per-head widths, scaled row blocks for prefill and greedy SM assignment for decode | `fastdc/kernels/` |

## Requirements

- Linux with an NVIDIA GPU of compute capability 8.0 or newer and a CUDA 13 driver.
- Python 3.12. The pinned versions are in `pyproject.toml` (PyTorch 2.13, Triton 3.7.1, transformers 5.17, vLLM 0.29.0).
- Hugging Face models with RoPE and standard multi-head or grouped-query attention: the Llama (including Yi),
  Mistral, Qwen2, Qwen3 and Phi-3 families. Gated models need a Hugging Face token (`hf auth login`).

## Installation

```bash
git clone https://github.com/Zeyu-ZEYU/FastDC.git && cd FastDC
curl -LsSf https://astral.sh/uv/install.sh | sh          # provides Python 3.12 if the system has none
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
pip install -e ".[vllm,benchmarks,test]"
```

The vLLM integration registers itself through the `vllm.general_plugins` entry point when the package is installed.

## Quick start

```bash
# offline stage: rotations and singular values (random tokens; whitened V basis from WikiText-2)
fastdc calibrate --model Qwen/Qwen3-8B --out calib/qwen3_8b.safetensors

# a FastDC plan from one energy removal rate r (or: --method palu --p 0.3 for the uniform pre-RoPE baseline)
fastdc plan --calibration calib/qwen3_8b.safetensors --method fastdc --r 0.02 --out plans/qwen3_8b_fastdc.safetensors

# generate with the reference engine and compare the stored KV cache with the uncompressed one
fastdc generate --model Qwen/Qwen3-8B --plan plans/qwen3_8b_fastdc.safetensors --prompt "Why is the sky blue?"
```

`python examples/quickstart.py --model Qwen/Qwen3-8B` runs the same steps in one script.

## Accuracy constraints: the F_p table

`fastdc sweep` builds the F_p table of a task: for a grid of energy removal rates r it builds the plan,
measures the accuracy on the task's calibration split (20% of the examples) and records the overall KV
compression rate. `fastdc select` returns the plan with the highest compression rate that keeps a given
share of the uncompressed accuracy.

```bash
fastdc sweep  --model Qwen/Qwen3-8B --calibration calib/qwen3_8b.safetensors --task gsm8k --out tables/gsm8k.json
fastdc select --table tables/gsm8k.json --threshold 0.99 --calibration calib/qwen3_8b.safetensors \
              --out plans/qwen3_8b_gsm8k_99.safetensors
fastdc evaluate --model Qwen/Qwen3-8B --plan plans/qwen3_8b_gsm8k_99.safetensors --task gsm8k   # evaluation split
```

Workloads (`fastdc/tasks.py`): `imdb` (genre classification, accuracy), `humaneval` (edit similarity), `gsm8k`
(accuracy), `arxiv` (summarization, ROUGE-2), `cocktail` (information retrieval, accuracy), `ruler`
(needle retrieval, accuracy), `scbench` (key-value retrieval in multi-turn mode, accuracy) and `pg19` (book
continuation, ROUGE-2), and the held-out workloads `mgsm`, `lcc`, `repobench` (next-line edit similarity), `bfcl`
(function calling) and `sharegpt` (multi-turn dialogue). Task options such as the number of examples or the context length are passed with
`--task-args '{"limit": 200, "context_tokens": 32768}'`.

The combined methods of the evaluation are options of the reference engine. `fastdc evaluate --kvquant`
stores the compressed K and V in 4 bits (KVQ-F, or KVQ-P with a `palu` plan), and `--keyformer 0.5` keeps
half of the tokens (KF-F, KF-P). `fastdc sweep --kvquant` and `fastdc sweep --keyformer` build their F_p
tables; with eviction, eviction and dimensionality compression contribute equally to the overall rate.

## Serving with vLLM

The plan is passed to vLLM's custom attention backend:

```bash
vllm serve Qwen/Qwen3-8B --attention-backend CUSTOM \
    --additional-config '{"fastdc": {"plan": "/abs/path/plans/qwen3_8b_fastdc.safetensors"}}'
```

When the model is loaded, the integration folds the V basis into the QKV and output projections. At run
time it rotates and truncates Q and K after RoPE, stores K and V at the per-head widths of the plan, and
runs the kernels of `fastdc/kernels`. The KV cache capacity that vLLM reports grows with the compression
rate. Plans of the `palu` method store the low-rank pre-RoPE K and reconstruct it before attention. Plans
of the `full` method run the same backend without compression. Continuous batching, chunked prefill,
prefix caching, piecewise CUDA graphs and tensor parallelism are supported. Options under `"fastdc"` (or
environment variables `FASTDC_<OPTION>`):

| Key | Default | Meaning |
|---|---|---|
| `plan` | (required) | compression plan written by `fastdc plan` or `fastdc select` |
| `balance` | `true` | kernel enhancement: scaled prefill row blocks and greedy SM assignment for decode; `false` uses fixed row blocks in natural order |
| `attention` | `kernels` | `kernels` (Triton) or `reference` (PyTorch) |
| `decompress_tokens` | `max(65536, max_model_len)` | `palu` only: tokens of reconstructed keys held at once |
| `profile` | (none) | JSON-lines file with the GPU time of the Q/K compression, the kernel scheduling and the key reconstruction per engine step |

The offline API takes the same options: `LLM(model, attention_backend="CUSTOM", additional_config={"fastdc": {...}})`,
as in `examples/vllm_generate.py`. The backend serves unquantized BF16 or FP16 models.

## Prefill-decode disaggregation

`fastdc/disagg` contains a vLLM KV connector that moves the cache pages of a request from a prefill instance
to a decode instance, and a proxy that runs the prompt on the prefill side and the generation on the decode
side. With a FastDC plan the pages hold the compressed K and V. To start a cluster:

```bash
python -m fastdc.disagg.launch --model Qwen/Qwen3-8B --plan plans/qwen3_8b_fastdc.safetensors \
    --prefill-gpus 0 --decode-gpus 1        # requests go to http://127.0.0.1:8000/v1/completions
```

`--bandwidth-gbps` limits the sending rate of the prefill side; `--smoke 4` sends four requests and stops.

## Benchmarks

- `scripts/functional_check.sh [GPU]` runs every component once at a small scale on one GPU: unit tests,
  offline stage, reference engine, F_p table, kernels, vLLM, serving harness and disaggregation.
- `benchmarks/serving/run.py` starts one vLLM instance per GPU group (`--gpus "0;1"` for two instances,
  `"0,1"` for one instance with tensor parallelism 2) behind a load balancer (`benchmarks/serving/router.py`,
  shortest token-based queue), replays a trace of a workload with Poisson arrivals and at most 16 requests in
  flight, and writes TTFT, TBT, JCT and throughput per method and arrival rate (`--rates inf` sends all
  requests at once). Every request generates the trace's output length with `ignore_eos`. Prefix caching is
  off unless `--prefix-caching` is given, and warm-up requests come from the task's calibration split.
  `benchmarks/serving/plot.py` plots the results.
- `benchmarks/kernels/attention_bench.py` times the attention kernels at full and compressed widths, with and
  without load balancing, and the construction of the schedules.

Every script documents its options in `--help`. Example:

```bash
fastdc calibrate --model Qwen/Qwen3-8B --out calib/qwen3_8b.safetensors
fastdc plan --calibration calib/qwen3_8b.safetensors --method fastdc --r 0.02 --out plans/fastdc.safetensors
fastdc plan --calibration calib/qwen3_8b.safetensors --method palu --p 0.3 --out plans/palu.safetensors
python benchmarks/serving/run.py --model Qwen/Qwen3-8B --gpus "0;1" --task arxiv --num-requests 200 --rates 0.5,1,2 \
       --method palu=plans/palu.safetensors --method fastdc=plans/fastdc.safetensors \
       --method fastdc_ke=plans/fastdc.safetensors:balance=false --out results/serving/arxiv
python benchmarks/serving/plot.py results/serving/arxiv --out serving.pdf
```

A method is `NAME` (vLLM's default backend) or `NAME=PLAN[:option=value,...]` with the options of the table
above.

## Tests

```bash
pytest tests                      # GPU tests are skipped when no GPU is visible
```

The vLLM tests start engines with `Qwen/Qwen3-0.6B` (override with `FASTDC_TEST_MODEL`) and cache their
calibration in `FASTDC_TEST_CACHE` (default `~/.cache/fastdc/tests`).

```bash
scripts/functional_check.sh 0     # all components once, on GPU 0
```

## Repository layout

```
fastdc/
  models.py         model access and a layer-by-layer runner for Hugging Face weights
  calibration.py    offline steps 1-2: post-RoPE Q/K capture, SVD per KV head, whitened V basis
  rates.py plan.py  offline step 3: energy rule (Eq. 1) and compression plans
  fp_table.py       F_p tables (accuracy constraint -> per-head rates)
  engine.py         reference engine (FastDC, Palu, quantization and eviction modifiers)
  tasks.py evaluate.py   workloads, metrics, 20/80 calibration/evaluation split
  kernels/          Triton kernels for the compressed paged cache and the load-balanced scheduling
  vllm_plugin/      vLLM attention backend, weight folding, per-layer cache layout
  disagg/           KV connector, proxy and launcher for prefill-decode disaggregation
  cli.py            the `fastdc` command
benchmarks/         serving (load balancer, load generator) and kernel benchmarks
examples/ scripts/ tests/
```

## Citation

```bibtex
@inproceedings{zhang2026fastdc,
  title     = {{FastDC}: Fast {KV} Dimensionality Compression for Decompression-Free and Attention-Accelerated {LLM} Inference},
  author    = {Zhang, Zeyu and Shen, Haiying},
  booktitle = {Proceedings of the 2026 ACM Symposium on Cloud Computing (SoCC '26)},
  year      = {2026}
}
```

## License

Apache License 2.0, see `LICENSE`.
