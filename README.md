## Repo of FastDC

Fast KV Dimensionality Compression for Efficient LLM Serving

## Datasets

### Accuracy Metric Selection

- For classification tasks and information retrieval tasks, we use the ***accuracy*** as the metric.
- For summarization tasks and article generation tasks, we use ***ROUGE-1*** [1] score as the accuracy score.
- For code completion, we use ***Edit Similarity (normalized Levenshtein distance)*** [2-3] as the accuracy.

### Dataset Dir
IMDb movie genre classification: /fastdc/datasets/imdb

arXiv summarization: /fastdc/datasets/arxiv

Cocktail for information retrieval: /fastdc/datasets/cocktail

HumanEval for code completion: /fastdc/datasets/humaneval

PG-19 for article generation: /fastdc/datasets/pg-19

## FastDC Dir
/fastdc  # FastDC related code
- datasets
- exp
- kernels
- measurements
- quantization
- svd_qkv

/vllm  # vLLM base code

```
datasets: it contains the datasets we use for validation.
exp: it has the implementation code of FastDC.
       # fastdc_plugin.py/keyformer(-f).py/kvquant(-f).py
kernels: kernel functions.
measurements: it has the measurement code for SVD and model analysis.
quantization: it has the code of quantization methods.
svd_qkv: SVD related code for QKV compression and analysis.
```

## References
[1] ROUGE Score, https://en.wikipedia.org/wiki/ROUGE_(metric)

[2] Zhang, Lei, et al. "Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs." arXiv preprint arXiv:2406.18294 (2024).

[3] String Similarity Metrics - Edit Distance, https://www.baeldung.com/cs/string-similarity-edit-distance
