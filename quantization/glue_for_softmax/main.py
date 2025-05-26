from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

dataset = load_dataset("glue", "mrpc")

print(dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
mapped_dataset = dataset.map(lambda x: tokenizer(x["sentence1"], x["sentence2"]), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
train_args = TrainingArguments("test-run-1")

trainer = Trainer(
    model=model,
    train_args=train_args,
    train_dataset=mapped_dataset["train"],
    eval_dataset=mapped_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
