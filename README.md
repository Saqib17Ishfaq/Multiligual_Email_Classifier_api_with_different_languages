# Multiligual_Email_Classifier_api_with_different_languages

# 📧 Email Spam Detection API

This is a Flask-based multilingual API that detects whether an email text is **SPAM** or **HAM (not spam)** using trained machine learning models.

---

## 🔧 Features Implemented

- ✅ Trained **three machine learning models**:
  - Logistic Regression
  - Naive Bayes (Multinomial)
  - Support Vector Machine (SVM with confidence using CalibratedClassifier)
  
- ✅ Created a **Flask API** to:
  - Accept email text via form-data
  - Return prediction (`SPAM` or `HAM`)
  - Include confidence score in percentage

- ✅ Implemented **input validation**:
  - Email must not exceed 100 words
  - Email must not contain digits
  - Must send input in correct list format or separated text

- ✅ Error handling for:
  - Missing input key
  - Invalid format
  - Validation failures
  - Server errors

- ✅ Added **logging** using Python’s `logging` module
  - All predictions are logged with timestamp, confidence, and result

- ✅ Multilingual Support:
  - Supports error/output messages in:
    - 🇬🇧 English (default)
    - 🇵🇰 Urdu
    - 🇹🇷 Turkish
    - 🇸🇦 Arabic
  - Set language using `Accept-Language` in request headers (e.g., `ar`, `ur`, `tr`, `en`)

---

## 🚀 How to Run the Project

1. Make sure you have Python 3.8+ installed.
2. Install dependencies (Flask, scikit-learn, pandas).
3. Train models using the training script and save to `all_models.pkl`.
4. Run the API using:
   
python app.py
