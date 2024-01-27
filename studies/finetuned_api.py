from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine-tuned-classification-model")
tokenizer = DistilBertTokenizer.from_pretrained("./fine-tuned-classification-model")

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

    # Get predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits).item()

    # Prepare the response
    response = {
        'text': text,
        'predicted_class': predicted_class,
        'class_probabilities': probabilities
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
