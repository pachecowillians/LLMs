import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)  # Binary classification, adjust num_labels based on your task

# Load the dataset (for simplicity, we'll use the IMDb dataset)
dataset = load_dataset("imdb")

# Tokenize the dataset
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_batch, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-classification-model")
tokenizer.save_pretrained("./fine-tuned-classification-model")
