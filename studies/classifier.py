from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier(["I am very happy!", "I am sad!"])

for result in results:
    print(result)