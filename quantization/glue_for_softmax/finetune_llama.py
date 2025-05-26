import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    LlamaConfig,
    Trainer,
    TrainingArguments,
)

model_name = "meta-llama/Llama-2-7b-chat-hf"
# configuration = LlamaConfig()
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
tokenizer.padding_side = "right"
# print(tokenizer.pad_token_id)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset("glue", "mrpc")


def map_func(x):
    return tokenizer(x["sentence1"], x["sentence2"])


mapped_dataset = dataset.map(map_func, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_args = TrainingArguments("llama-2", per_device_train_batch_size=2, per_device_eval_batch_size=2)
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=mapped_dataset["train"],
    eval_dataset=mapped_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()


logits, labels, _ = trainer.predict(mapped_dataset["test"])
predictions = np.argmax(logits, axis=-1)
all = np.sum(predictions == labels)
print(all / len(labels))
