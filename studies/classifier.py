from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier(["I am very happy!", "I am not sad!"])

print(result)