# SvaraAI – AI/ML Engineer Internship Assignment

This repository contains my solution for the SvaraAI internship assignment (22-09-2025 to 24-09-2025).  
The goal is to classify email replies into **positive**, **negative**, or **neutral**, deploy the best model as an API, and provide reasoning on design decisions.  

---

## 🚀 Part A – ML/NLP Pipeline

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

---

## 🌐 Deployment

- Implemented a **Flask API** (`app.py`) with `/predict` endpoint.  
- **Input:**  
  ```json
  { "text": "Looking forward to the demo!" }
- **Output:**
  ```json
  { "label": "positive", "confidence": 0.87 }




