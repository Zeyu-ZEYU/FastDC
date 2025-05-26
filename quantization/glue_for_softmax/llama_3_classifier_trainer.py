import os
import warnings

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
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
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    IntervalStrategy,
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


filename = "./financial_news.csv"


df = pd.read_csv(filename, names=["sentiment", "text"], encoding="utf-8", encoding_errors="replace")

X_train = list()
X_test = list()
for sentiment in ["positive", "neutral", "negative"]:
    train, test = train_test_split(df[df.sentiment == sentiment], train_size=300, test_size=300, random_state=42)
    X_train.append(train)
    X_test.append(test)

X_train = pd.concat(X_train).sample(frac=1, random_state=10)
X_test = pd.concat(X_test)

eval_idx = [idx for idx in df.index if idx not in list(X_train.index) + list(X_test.index)]
X_eval = df[df.index.isin(eval_idx)]
X_eval = X_eval.groupby("sentiment", group_keys=False).apply(lambda x: x.sample(n=50, random_state=10, replace=True))
X_train = X_train.reset_index(drop=True)


def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets,
            determine if it is positive, neutral, or negative, and return the answer as
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()


def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets,
            determine if it is positive, neutral, or negative, and return the answer as
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = """.strip()


X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["text"])

y_true = X_test.sentiment
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)


def evaluate(y_true, y_pred):
    labels = ["positive", "neutral", "negative"]
    mapping = {"positive": 2, "neutral": 1, "none": 1, "negative": 0}

    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy for label {label}: {accuracy:.3f}")

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print("\nClassification Report:")
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print("\nConfusion Matrix:")
    print(conf_matrix)


model_name = "/home/zeyu/large_files/metaresearch/llama-3/transformers/8b-chat-hf/1"


compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 512  # 2048
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
tokenizer.pad_token_id = tokenizer.eos_token_id


def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1,
            temperature=0.0,
        )
        result = pipe(prompt)
        answer = result[0]["generated_text"].split("=")[-1]
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred


# y_pred = predict(test, model, tokenizer)

# evaluate(y_true, y_pred)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


output_dir = "/home/zeyu/large_files/llama_3_trained_weigths"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

training_arguments = TrainingArguments(
    output_dir=output_dir,  # directory to save and repository id
    num_train_epochs=5,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,  # log every 10 steps
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",  # use cosine learning rate scheduler
    report_to="tensorboard",  # report metrics to tensorboard
    # evaluation_strategy="steps",              # save checkpoint every epoch
    # load_best_model_at_end = True,
    # eval_steps = 25,
    # metric_for_best_model = 'accuracy',
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    # eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    # compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)


trainer.train()
# Save trained model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(output_dir)
