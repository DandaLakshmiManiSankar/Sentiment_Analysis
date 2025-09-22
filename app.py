
from flask import Flask, request, render_template_string
from pyngrok import ngrok
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import re

# ---------------------------
# Model setup
# ---------------------------
model_path = "distibert_model"  # folder containing pytorch_model.safetensors

# Load tokenizer & fine-tuned model
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Simple text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

html_form = """
<!doctype html>
<title>Email Sentiment Classification</title>
<h2>Enter an email reply:</h2>
<form action="/predict_web" method="post">
  <textarea name="text" rows="4" cols="50"></textarea><br><br>
  <input type="submit" value="Predict">
</form>
{% if result %}
<h3>Prediction:</h3>
<p>Label: <b>{{ result.label }}</b></p>
<p>Confidence: <b>{{ result.confidence }}</b></p>
{% endif %}
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_form)

@app.route("/predict_web", methods=["POST"])
def predict_web():
    text = request.form.get("text", "")
    if not text:
        return render_template_string(html_form, result={"label": "N/A", "confidence": 0})
    
    text_clean = clean_text(text)
    inputs = tokenizer(text_clean, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(torch.argmax(logits, dim=1).item())

    result = {"label": label_map[pred_idx], "confidence": float(probs[pred_idx])}
    return render_template_string(html_form, result=result)

# ---------------------------
# Start ngrok tunnel
# ---------------------------
public_url = ngrok.connect(5000)
print("Public URL:", public_url)

# ---------------------------
# Run Flask app
# ---------------------------
app.run(port=5000)
