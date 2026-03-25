import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from preprocessing import clean_text

print("Script started")

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
data_path = os.path.join(BASE_DIR, 'data', 'sample_tickets.csv')
df = pd.read_csv(data_path)

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

pickle.dump(model, open(model_path, 'wb'))
pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

print("Model trained and saved!")