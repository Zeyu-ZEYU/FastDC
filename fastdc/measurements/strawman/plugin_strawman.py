import os
import warnings

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import SFTTrainer, setup_chat_format

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


filename = "../data/train_data.txt"
test_file_name = "../data/test_data_solution.txt"
# test_file_name = "./data/train_data.txt"


def strip_whitespace(x):
    return x.strip() if type(x) is str else x


df = pd.read_csv(
    filename,
    sep=":::",
    names=["id", "title", "genre", "desc"],
    converters={
        "id": strip_whitespace,
        "title": strip_whitespace,
        "genre": strip_whitespace,
        "desc": strip_whitespace,
    },
    encoding="utf-8",
    encoding_errors="replace",
)
df_test = pd.read_csv(
    test_file_name,
    sep=":::",
    names=["id", "title", "genre", "desc"],
    converters={
        "id": strip_whitespace,
        "title": strip_whitespace,
        "genre": strip_whitespace,
        "desc": strip_whitespace,
    },
    encoding="utf-8",
    encoding_errors="replace",
)

labels = set(df.genre)

label_mapping = {
    "sport": 0,
    "action": 1,
    "history": 2,
    "family": 3,
    "talk-show": 4,
    "adult": 5,
    "sci-fi": 6,
    "comedy": 7,
    "game-show": 8,
    "romance": 9,
    "fantasy": 10,
    "animation": 11,
    "thriller": 12,
    "horror": 13,
    "reality-tv": 14,
    "news": 15,
    "musical": 16,
    "short": 17,
    "biography": 18,
    "crime": 19,
    "mystery": 20,
    "western": 21,
    "music": 22,
    "drama": 23,
    "documentary": 24,
    "adventure": 25,
    "war": 26,
}


df_train = df.sample(frac=1.0, random_state=10).reset_index(drop=True)
df_test = df_test.sample(frac=0.2, random_state=10).reset_index(drop=True)


def create_train_input(data_point):
    return f"{data_point["title"]}: {data_point["desc"]}. Genre={data_point["genre"]}"


train_in_new = pd.DataFrame(
    df_train.apply(create_train_input, axis=1), columns=["text"]
)
ids = df_train.apply(lambda x: label_mapping.get(x["genre"]), axis=1)
train_in_new["label"] = ids
df_train = train_in_new
ds_train = Dataset.from_pandas(df_train)

test_in_new = pd.DataFrame(df_test.apply(create_train_input, axis=1), columns=["text"])
test_in_new["label"] = df_test.apply(lambda x: label_mapping.get(x["genre"]), axis=1)
df_test = test_in_new
ds_test = Dataset.from_pandas(df_test)


class_weights = (1 / df_train.label.value_counts(normalize=True).sort_index()).tolist()
class_weights = torch.tensor(class_weights)
class_weights = class_weights / class_weights.sum()


### prepare collator
model_name = "/home/ubuntu/checkpoint-30000"


MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


def llama_preprocessing_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)


tokenized_datasets = ds_train.map(llama_preprocessing_function, batched=True)
tokenized_datasets.set_format("torch")
tokenized_test_datasets = ds_test.map(llama_preprocessing_function, batched=True)
tokenized_test_datasets.set_format("torch")


collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.args.device
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get("logits")

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    from sklearn.metrics import top_k_accuracy_score

    # softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    # probabilities = softmax(predictions)
    top_1_accuracy = top_k_accuracy_score(labels, predictions, k=1)
    top_5_accuracy = top_k_accuracy_score(labels, predictions, k=5)
    top_10_accuracy = top_k_accuracy_score(labels, predictions, k=10)

    try:
        # it's a classification task, take the argmax
        predictions_processed = np.argmax(predictions, axis=1)

        # Calculate Pearson correlation
        pearson, _ = pearsonr(labels, predictions_processed)
        accuracy = accuracy_score(labels, predictions_processed)
        class_report = classification_report(labels, predictions_processed, digits=5)
        # confuse_mat = confusion_matrix(labels, predictions_processed)
        f1 = f1_score(labels, predictions_processed, average="micro")
        precision = precision_score(labels, predictions_processed, average="micro")
        recall = recall_score(labels, predictions_processed, average="micro")

        print(class_report)

        print("top_1_accuracy:", top_1_accuracy)
        print("top_5_accuracy:", top_5_accuracy)
        print("top_10_accuracy:", top_10_accuracy)

        return {
            "pearson": pearson,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {"None": None}


### load model


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
)


lora_config = LoraConfig(
    r=16,  # the dimension of the low-rank matrices
    lora_alpha=8,  # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,  # dropout probability of the LoRA layers
    bias="none",  # wether to train bias weights, set to 'none' for attention layers
    task_type="SEQ_CLS",
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=len(labels),
    device_map=device,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# for n, p in model.named_parameters():
#     print(n)
# exit()

# del model.score.original_module


training_args = TrainingArguments(
    output_dir="./seqcls_eval_results",
    per_device_eval_batch_size=16,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
eval_results = trainer.evaluate(eval_dataset=tokenized_test_datasets)

print(eval_results)
