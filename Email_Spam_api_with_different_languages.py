


# # this code is training all the models on the dataset 

# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split

# # === Step 1: Load Dataset ===
# data_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\spam_ham_dataset.csv"
# data = pd.read_csv(data_path)

# # Adjust column names as needed (update this if your dataset is different)
# data = data[['label', 'text']]
# data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # 0 = ham, 1 = spam

# # === Step 2: Split Dataset ===
# X_train, X_test, y_train, y_test = train_test_split(
#     data['text'], data['label'], test_size=0.2, random_state=42
# )

# # === Step 3: Vectorize Text ===
# vectorizer = TfidfVectorizer(stop_words='english')
# X_train_vec = vectorizer.fit_transform(X_train)

# # === Step 4: Train Models ===
# logreg = LogisticRegression()
# logreg.fit(X_train_vec, y_train)

# nb_model = MultinomialNB()
# nb_model.fit(X_train_vec, y_train)

# svm_model = LinearSVC()
# svm_model.fit(X_train_vec, y_train)

# # === Step 5: Save All Models ===
# assets = {
#     'logreg': logreg,
#     'nb_model': nb_model,
#     'svm_model': svm_model,
#     'vectorizer': vectorizer
# }

# with open(r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl", 'wb') as f:
#     pickle.dump(assets, f)

# print("✅ All models trained and saved successfully!")


#================================this api is predictiing through nb model===================================
# from flask import Flask, request, jsonify
# import pickle
# import logging
# import re
# import ast
# from datetime import datetime

# # === Setup Logging ===
# logging.basicConfig(
#     filename='email_api.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # === Load Model and Vectorizer ===
# model_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl"

# try:
#     with open(model_path, 'rb') as f:
#         assets = pickle.load(f)
#         model = assets['nb_model']  # Change to 'logreg' or 'svm_model' if needed
#         vectorizer = assets['vectorizer']
# except Exception as e:
#     logging.critical(f"Model loading failed: {e}")
#     raise

# app = Flask(__name__)

# # === Helper: Validate email text ===
# def is_valid_email_text(text):
#     if len(text.split()) > 100:
#         return False, "Email text exceeds 100 words"
#     if re.search(r'\d', text):  # block any digits
#         return False, "Email text contains digits which are not allowed"
#     return True, ""

# # === Predict Endpoint ===
# @app.route('/predict', methods=['POST'])
# def predict():
#     logging.info("🔔 /predict endpoint hit at %s from IP: %s",
#                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                   request.remote_addr)

#     try:
#         # ✅ Check key existence
#         if 'email_value' not in request.form:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing Field",
#                 "message": "'email_value' key is required in form-data."
#             }), 400

#         raw_input = request.form.get('email_value')
#         if not raw_input:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing email_value",
#                 "message": "The email_value field must not be empty."
#             }), 400

#         # ✅ Parse list input (safe)
#         if raw_input.strip().startswith("[") and raw_input.strip().endswith("]"):
#             try:
#                 email_list = ast.literal_eval(raw_input)
#                 if not isinstance(email_list, list):
#                     raise ValueError
#             except:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Invalid Format",
#                     "message": "Could not parse email_value as a list."
#                 }), 400
#         else:
#             # fallback: split by common delimiters
#             for splitter in ['||', '\n', ';', ',']:
#                 if splitter in raw_input:
#                     email_list = [e.strip() for e in raw_input.split(splitter)]
#                     break
#             else:
#                 email_list = [raw_input.strip()]

#         # ✅ Filter out empty strings
#         email_list = [e for e in email_list if e]
#         if not email_list:
#             return jsonify({
#                 "code": 400,
#                 "error": "Empty Input",
#                 "message": "No valid email texts provided."
#             }), 400

#         predictions = []

#         for i, email_text in enumerate(email_list):
#             valid, reason = is_valid_email_text(email_text)
#             if not valid:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Validation Error",
#                     "message": f"Email {i + 1} failed validation: {reason}"
#                 }), 400

#             email_vec = vectorizer.transform([email_text])
#             prediction = model.predict(email_vec)[0]

#             if hasattr(model, "predict_proba"):
#                 probas = model.predict_proba(email_vec)[0]
#                 confidence = round(probas[prediction] * 100, 2)
#             else:
#                 confidence = None

#             result = "SPAM" if prediction == 1 else "HAM"
#             conf_str = f"{confidence}%" if confidence is not None else "N/A"
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_line = f"[{timestamp}] Prediction: {result} | Confidence: {conf_str} | Text: {email_text[:50]}..."

#             # ✅ Log to file
#             logging.info(log_line)

#             # ✅ Also return in JSON
#             predictions.append({
#                 "email": email_text,
#                 "prediction": result,
#                 "confidence": conf_str,
#                 "log": log_line
#             })

#         return jsonify({"predictions": predictions})

#     except Exception as e:
#         logging.error(f"Prediction failed: {str(e)}")
#         return jsonify({
#             "code": 500,
#             "error": "Internal Server Error",
#             "message": str(e)
#         }), 500

# # === Start Flask App ===
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

#================================this api is predictiing through logreg model===================================
# from flask import Flask, request, jsonify
# import pickle
# import logging
# import re
# import ast
# from datetime import datetime

# # === Setup Logging ===
# logging.basicConfig(
#     filename='email_api.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # === Load Logistic Regression Model and Vectorizer ===
# model_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl"

# try:
#     with open(model_path, 'rb') as f:
#         assets = pickle.load(f)
#         model = assets['logreg']  # ✅ Logistic Regression
#         vectorizer = assets['vectorizer']
# except Exception as e:
#     logging.critical(f"Model loading failed: {e}")
#     raise

# app = Flask(__name__)

# # === Helper ===
# def is_valid_email_text(text):
#     if len(text.split()) > 100:
#         return False, "Email text exceeds 100 words"
#     if re.search(r'\d', text):
#         return False, "Email text contains digits which are not allowed"
#     return True, ""

# # === Predict Endpoint ===
# @app.route('/predict', methods=['POST'])
# def predict():
#     logging.info("🔔 /predict endpoint hit at %s from IP: %s",
#                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                   request.remote_addr)

#     try:
#         if 'email_value' not in request.form:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing Field",
#                 "message": "'email_value' key is required in form-data."
#             }), 400

#         raw_input = request.form.get('email_value')
#         if not raw_input:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing email_value",
#                 "message": "The email_value field must not be empty."
#             }), 400

#         if raw_input.strip().startswith("[") and raw_input.strip().endswith("]"):
#             try:
#                 email_list = ast.literal_eval(raw_input)
#                 if not isinstance(email_list, list):
#                     raise ValueError
#             except:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Invalid Format",
#                     "message": "Could not parse email_value as a list."
#                 }), 400
#         else:
#             for splitter in ['||', '\n', ';', ',']:
#                 if splitter in raw_input:
#                     email_list = [e.strip() for e in raw_input.split(splitter)]
#                     break
#             else:
#                 email_list = [raw_input.strip()]

#         email_list = [e for e in email_list if e]
#         if not email_list:
#             return jsonify({
#                 "code": 400,
#                 "error": "Empty Input",
#                 "message": "No valid email texts provided."
#             }), 400

#         predictions = []

#         for i, email_text in enumerate(email_list):
#             valid, reason = is_valid_email_text(email_text)
#             if not valid:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Validation Error",
#                     "message": f"Email {i + 1} failed validation: {reason}"
#                 }), 400

#             email_vec = vectorizer.transform([email_text])
#             prediction = model.predict(email_vec)[0]
#             probas = model.predict_proba(email_vec)[0]
#             confidence = round(probas[prediction] * 100, 2)

#             result = "SPAM" if prediction == 1 else "HAM"
#             conf_str = f"{confidence}%"
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_line = f"[{timestamp}] Prediction: {result} | Confidence: {conf_str} | Text: {email_text[:50]}..."

#             logging.info(log_line)

#             predictions.append({
#                 "email": email_text,
#                 "prediction": result,
#                 "confidence": conf_str,
#                 "log": log_line
#             })

#         return jsonify({"predictions": predictions})

#     except Exception as e:
#         logging.error(f"Prediction failed: {str(e)}")
#         return jsonify({
#             "code": 500,
#             "error": "Internal Server Error",
#             "message": str(e)
#         }), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)




#======================it is for training of svm model=================================== 
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.model_selection import train_test_split

# # === Step 1: Load Dataset ===
# data_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\spam_ham_dataset.csv"
# data = pd.read_csv(data_path)

# # Adjust column names as needed (update this if your dataset is different)
# data = data[['label', 'text']]
# data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # 0 = ham, 1 = spam

# # === Step 2: Split Dataset ===
# X_train, X_test, y_train, y_test = train_test_split(
#     data['text'], data['label'], test_size=0.2, random_state=42
# )

# # === Step 3: Vectorize Text ===
# vectorizer = TfidfVectorizer(stop_words='english')
# X_train_vec = vectorizer.fit_transform(X_train)

# # === Step 4: Train Models ===
# logreg = LogisticRegression()
# logreg.fit(X_train_vec, y_train)

# nb_model = MultinomialNB()
# nb_model.fit(X_train_vec, y_train)

# # ✅ Calibrated SVM to enable predict_proba
# base_svm = LinearSVC()
# svm_model = CalibratedClassifierCV(base_svm, cv=5)
# svm_model.fit(X_train_vec, y_train)

# # === Step 5: Save All Models ===
# assets = {
#     'logreg': logreg,
#     'nb_model': nb_model,
#     'svm_model': svm_model,
#     'vectorizer': vectorizer
# }

# with open(r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl", 'wb') as f:
#     pickle.dump(assets, f)

# print("\u2705 All models trained and saved successfully!")




#================================this api is predictiing through svm model=================================== 
# from flask import Flask, request, jsonify
# import pickle
# import logging
# import re
# import ast
# from datetime import datetime

# # === Setup Logging ===
# logging.basicConfig(
#     filename='email_api.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # === Load SVM Model with Probability Support and Vectorizer ===
# model_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl"

# try:
#     with open(model_path, 'rb') as f:
#         assets = pickle.load(f)
#         model = assets['svm_model']  # ✅ SVM with CalibratedClassifierCV
#         vectorizer = assets['vectorizer']
# except Exception as e:
#     logging.critical(f"Model loading failed: {e}")
#     raise

# app = Flask(__name__)

# # === Helper: Validate Email Text ===
# def is_valid_email_text(text):
#     if len(text.split()) > 100:
#         return False, "Email text exceeds 100 words"
#     if re.search(r'\d', text):
#         return False, "Email text contains digits which are not allowed"
#     return True, ""

# # === Predict Endpoint ===
# @app.route('/predict', methods=['POST'])
# def predict():
#     logging.info("🔔 /predict endpoint hit at %s from IP: %s",
#                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                   request.remote_addr)

#     try:
#         if 'email_value' not in request.form:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing Field",
#                 "message": "'email_value' key is required in form-data."
#             }), 400

#         raw_input = request.form.get('email_value')
#         if not raw_input:
#             return jsonify({
#                 "code": 400,
#                 "error": "Missing email_value",
#                 "message": "The email_value field must not be empty."
#             }), 400

#         if raw_input.strip().startswith("[") and raw_input.strip().endswith("]"):
#             try:
#                 email_list = ast.literal_eval(raw_input)
#                 if not isinstance(email_list, list):
#                     raise ValueError
#             except:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Invalid Format",
#                     "message": "Could not parse email_value as a list."
#                 }), 400
#         else:
#             for splitter in ['||', '\n', ';', ',']:
#                 if splitter in raw_input:
#                     email_list = [e.strip() for e in raw_input.split(splitter)]
#                     break
#             else:
#                 email_list = [raw_input.strip()]

#         email_list = [e for e in email_list if e]
#         if not email_list:
#             return jsonify({
#                 "code": 400,
#                 "error": "Empty Input",
#                 "message": "No valid email texts provided."
#             }), 400

#         predictions = []

#         for i, email_text in enumerate(email_list):
#             valid, reason = is_valid_email_text(email_text)
#             if not valid:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Validation Error",
#                     "message": f"Email {i + 1} failed validation: {reason}"
#                 }), 400

#             email_vec = vectorizer.transform([email_text])
#             prediction = model.predict(email_vec)[0]

#             if hasattr(model, "predict_proba"):
#                 probas = model.predict_proba(email_vec)[0]
#                 confidence = round(probas[prediction] * 100, 2)
#             else:
#                 confidence = None

#             result = "SPAM" if prediction == 1 else "HAM"
#             conf_str = f"{confidence}%" if confidence is not None else "N/A"
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_line = f"[{timestamp}] Prediction: {result} | Confidence: {conf_str} | Text: {email_text[:50]}..."

#             logging.info(log_line)

#             predictions.append({
#                 "email": email_text,
#                 "prediction": result,
#                 "confidence": conf_str,
#                 "log": log_line
#             })

#         return jsonify({"predictions": predictions})

#     except Exception as e:
#         logging.error(f"Prediction failed: {str(e)}")
#         return jsonify({
#             "code": 500,
#             "error": "Internal Server Error",
#             "message": str(e)
#         }), 500

# # === Run the App ===
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)



#================================this api is predictiing through all models and with different languages ===================================
# from flask import Flask, request, jsonify
# import pickle
# import logging
# import re
# import ast
# from datetime import datetime

# # === Setup Logging ===
# logging.basicConfig(
#     filename='email_api.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # === Load All Models ===
# model_path = r"C:\Users\Yasir\Desktop\Internship\Email Spam Detection\all_models.pkl"

# try:
#     with open(model_path, 'rb') as f:
#         assets = pickle.load(f)
# except Exception as e:
#     logging.critical(f"Model loading failed: {e}")
#     raise

# # === Language Translations ===
# translations = {
#     'en': {
#         'missing_field': "'email_value' key is required in form-data.",
#         'missing_value': "The email_value field must not be empty.",
#         'invalid_format': "Could not parse email_value as a list.",
#         'empty_input': "No valid email texts provided.",
#         'validation_error': "Email {i} failed validation: {reason}",
#         'digit_error': "Email text contains digits which are not allowed",
#         'length_error': "Email text exceeds 100 words",
#         'internal_error': "Internal Server Error",
#         'spam': "SPAM",
#         'ham': "HAM"
#     },
#     'ur': {
#         'missing_field': "'email_value' فارم ڈیٹا میں لازمی ہے۔",
#         'missing_value': "email_value فیلڈ خالی نہیں ہونی چاہیے۔",
#         'invalid_format': "email_value کو فہرست کے طور پر نہیں پڑھا جا سکا۔",
#         'empty_input': "کوئی درست ای میل فراہم نہیں کی گئی۔",
#         'validation_error': "ای میل {i} کی توثیق ناکام ہوئی: {reason}",
#         'digit_error': "ای میل میں نمبرز شامل ہیں جو قابل قبول نہیں۔",
#         'length_error': "ای میل 100 سے زیادہ الفاظ پر مشتمل ہے۔",
#         'internal_error': "داخلی سرور کی خرابی",
#         'spam': "سپیم",
#         'ham': "محفوظ"
#     },
#     'tr': {
#         'missing_field': "'email_value' form verisinde gerekli.",
#         'missing_value': "email_value alanı boş olmamalı.",
#         'invalid_format': "email_value listesi ayrıştırılamadı.",
#         'empty_input': "Geçerli e-posta metni sağlanmadı.",
#         'validation_error': "E-posta {i} doğrulama hatası: {reason}",
#         'digit_error': "E-posta metni sayılar içeriyor, izin verilmez.",
#         'length_error': "E-posta metni 100 kelimeyi aşıyor.",
#         'internal_error': "Dahili Sunucu Hatası",
#         'spam': "SPAM",
#         'ham': "HAM"
#     },
#     'ar': {
#         'missing_field': "مطلوب وجود المفتاح 'email_value' في بيانات النموذج.",
#         'missing_value': "لا يمكن أن يكون حقل email_value فارغًا.",
#         'invalid_format': "تعذر تحليل email_value كقائمة.",
#         'empty_input': "لم يتم توفير نصوص بريد إلكتروني صالحة.",
#         'validation_error': "فشل التحقق من البريد الإلكتروني {i}: {reason}",
#         'digit_error': "نص البريد الإلكتروني يحتوي على أرقام غير مسموح بها.",
#         'length_error': "نص البريد الإلكتروني يتجاوز 100 كلمة.",
#         'internal_error': "خطأ في الخادم الداخلي",
#         'spam': "بريد مزعج",
#         'ham': "بريد آمن"
#     }
# }

# # === Flask App Init ===
# app = Flask(__name__)

# def get_locale():
#     lang = request.headers.get("Accept-Language", "en")
#     return lang.split(',')[0][:2] if lang else 'en'

# def tr(key, **kwargs):
#     lang = get_locale()
#     return translations.get(lang, translations['en'])[key].format(**kwargs)

# def is_valid_email_text(text):
#     if len(text.split()) > 100:
#         return False, 'length_error'
#     if re.search(r'\d', text):
#         return False, 'digit_error'
#     return True, ""

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Choose model based on a header (default to SVM)
#         model_type = request.form.get('model_type', 'logreg')
#         if model_type not in assets:
#             model_type = 'logreg'
#         model = assets[model_type]
#         vectorizer = assets['vectorizer']

#         if 'email_value' not in request.form:
#             return jsonify({"code": 400, "error": "Missing Field", "message": tr('missing_field')}), 400

#         raw_input = request.form.get('email_value')
#         if not raw_input:
#             return jsonify({"code": 400, "error": "Missing email_value", "message": tr('missing_value')}), 400

#         if raw_input.strip().startswith("[") and raw_input.strip().endswith("]"):
#             try:
#                 email_list = ast.literal_eval(raw_input)
#                 if not isinstance(email_list, list):
#                     raise ValueError
#             except:
#                 return jsonify({"code": 400, "error": "Invalid Format", "message": tr('invalid_format')}), 400
#         else:
#             for splitter in ['||', '\n', ';', ',']:
#                 if splitter in raw_input:
#                     email_list = [e.strip() for e in raw_input.split(splitter)]
#                     break
#             else:
#                 email_list = [raw_input.strip()]

#         email_list = [e for e in email_list if e]
#         if not email_list:
#             return jsonify({"code": 400, "error": "Empty Input", "message": tr('empty_input')}), 400

#         predictions = []
#         for i, email_text in enumerate(email_list):
#             valid, reason_key = is_valid_email_text(email_text)
#             if not valid:
#                 return jsonify({
#                     "code": 400,
#                     "error": "Validation Error",
#                     "message": tr('validation_error', i=i+1, reason=tr(reason_key))
#                 }), 400

#             email_vec = vectorizer.transform([email_text])
#             prediction = model.predict(email_vec)[0]

#             if hasattr(model, "predict_proba"):
#                 probas = model.predict_proba(email_vec)[0]
#                 confidence = round(probas[prediction] * 100, 2)
#             else:
#                 confidence = None

#             result = tr('spam') if prediction == 1 else tr('ham')
#             conf_str = f"{confidence}%" if confidence is not None else "N/A"

#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_line = f"[{timestamp}] Prediction: {result} | Confidence: {conf_str} | Text: {email_text[:50]}..."
#             logging.info(log_line)

#             predictions.append({
#                 "email": email_text,
#                 "prediction": result,
#                 "confidence": conf_str,
#                 "log": log_line
#             })

#         return jsonify({"predictions": predictions})

#     except Exception as e:
#         logging.error(f"Prediction failed: {str(e)}")
#         return jsonify({
#             "code": 500,
#             "error": "Internal Server Error",
#             "message": tr('internal_error')
#         }), 500

# # === Run App ===
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)



