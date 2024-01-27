import torch
import numpy as np
import evaluate
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []

    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)
    
    return texts, labels

# https://ai.stanford.edu/~amaas/data/sentiment/

train_texts, train_labels = read_imdb_split("./studies/aclImdb/train")
test_texts, test_labels = read_imdb_split("./studies/aclImdb/test")

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.labels)
    
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = IMDBDataset(train_encodings, train_labels)
val_dataset = IMDBDataset(val_encodings, val_labels)
test_dataset = IMDBDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32, 
    # gradient_accumulation_steps=4,
    warmup_steps=200,
    learning_rate=1e-5,
    weight_decay=0.01,
    # logging_dir="./logs",
    # logging_steps=10,
    # evaluation_strategy="epoch",
)

model = DistilBertForSequenceClassification.from_pretrained(model_name)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model.to(device)

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=compute_metrics
)

trainer.train()