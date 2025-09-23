# SvaraAI – AI/ML Engineer Internship Assignment

This repository contains my solution for the SvaraAI internship assignment (22-09-2025 to 24-09-2025).  
The goal is to classify email replies into **positive**, **negative**, or **neutral**, deploy the best model as an API, and provide reasoning on design decisions.  

---

## 🚀 ML/NLP Pipeline

1. **Preprocessing**  
   - Cleaned text (lowercasing, removing special chars).  
   - Handled missing values.  

2. **Models**  
   - **Baseline:** Logistic Regression on TF-IDF features.  
   - **Transformer:** Fine-tuned `distilbert-base-uncased` using Hugging Face.  

3. **Evaluation**  
   - Metrics: Accuracy, F1, and Classification Report.  
   - Compared baseline vs transformer.  
   - Final decision: DistilBERT chosen for production (better balance between performance and generalization).
  
| Model               | Accuracy                      | Weighted F1 | Notes                                                                                                  |
| ------------------- | ----------------------------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| Logistic Regression | \~0.80–0.95 (depends on data) | \~0.80–0.95 | Simple, fast, interpretable. Works well on small datasets.                                             |
| DistilBERT          | \~0.85–1.0                    | \~0.85–1.0  | Powerful transformer, captures semantic meaning. Handles complex sentences. Slower to train and serve. |
  

My Public Drive Video Explanation Link : https://drive.google.com/file/d/1DNsA7H-F64RSOtcPHpHo_HMNTGbH3Ym5/view?usp=drive_link

---

## 🌐 Deployment

- Implemented a **Flask API** (`app.py`) with `/predict` endpoint.  
- **Input:**  
  ```json
  { "text": "Looking forward to the demo!" }
- **Output:**
  ```json
  { "label": "positive", "confidence": 0.87 }

## Project Setup (In Google Colab Environment)

step-1 : git clone https://github.com/DandaLakshmiManiSankar/Sentiment_Analysis.git

step-2 : cd Sentiment_Analysis

step-3 : !pip install -r requirements.txt

step-4 : !python app.py
