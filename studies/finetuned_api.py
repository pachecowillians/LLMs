from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Define the classification endpoint
@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json(force=True)
    text = data['text']

    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    predicted_class_index = torch.argmax(outputs.logits).item()
    
    # Map predicted class index to label
    predicted_label = "positive" if predicted_class_index == 1 else "negative"

    # Get predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]

    # Prepare the response
    response = {
        'text': text,
        'predicted_class': predicted_label,
        'class_probabilities': probabilities
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
