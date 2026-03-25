import streamlit as st
import pickle
import os
from src.preprocessing import clean_text

# Title
st.title("Ticket Classification System")

st.write("Enter a customer support ticket below:")

# Input box
user_input = st.text_area("Enter Ticket Text")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Prediction function
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

# Button
if st.button("Predict"):
    if user_input:
        category, priority = predict_ticket(user_input)

        st.success(f"Category: {category}")
        st.warning(f"Priority: {priority}")
    else:
        st.error("Please enter some text")