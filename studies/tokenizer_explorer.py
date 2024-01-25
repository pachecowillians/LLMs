from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

results = classifier(["I am very happy!", "I am sad!"])

for result in results:
    print(result)

X_train = ["I am sad", "I am happy"]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)

    labels = torch.argmax(predictions, dim=1)
    print(labels)