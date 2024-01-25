import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
