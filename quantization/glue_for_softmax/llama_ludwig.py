import logging
import os

import yaml
from datasets import load_dataset
from ludwig.api import LudwigModel

config_str = """
model_type: llm
base_model: meta-llama/Llama-2-7b-hf


adapter:
  type: lora

prompt:
  template: |
    ### Instruction:
    Is sentence1 semantically equivalent to sentence2? Output 1 if they are equivalent and 0 if they are not.

    ### Sentence1:
    {sentence1}

    ### Sentence2:
    {sentence2}

    ### Response:

input_features:
  - name: prompt
    type: text
    preprocessing:
      max_sequence_length: 256

output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 4

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 2
  gradient_accumulation_steps: 16
  epochs: 1
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1

backend:
  type: local
"""

config = yaml.safe_load(config_str)
raw_datasets = load_dataset("glue", "mrpc")
raw_datasets["train"].to_csv("./glue_train.csv")
raw_datasets["validation"].to_csv("./glue_validation.csv")
raw_datasets["test"].to_csv("./glue_test.csv")

model = LudwigModel(config=config, logging_level=logging.INFO)
