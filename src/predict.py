import pickle
import os
from preprocessing import clean_text

print("Prediction script started")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

print("Model path:", model_path)

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def predict_ticket(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    category = model.predict(vector)[0]

    if "urgent" in clean or "not working" in clean:
        priority = "High"
    elif "slow" in clean:
        priority = "Medium"
    else:
        priority = "Low"

    return category, priority

print(predict_ticket("Payment not working urgently"))