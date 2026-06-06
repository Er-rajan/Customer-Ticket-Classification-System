# Customer Support Ticket Classification System

## Overview
This project automatically classifies customer support tickets and assigns priority levels using NLP and Machine Learning.

## Features
- Text preprocessing & cleaning
- Ticket category classification
- Priority tagging (High / Medium / Low)
- Model evaluation

## Tech Stack
- Python
- NLTK / spaCy
- Scikit-learn
- Jupyter Notebook

## Project Structure
ticket-classification-system/
│
├── data/
│   └── sample_tickets.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore


## How to Run the Project

### 1-> Clone the Repository

```bash
git clone https://github.com/Er-rajan/ticket-classification-system.git
cd ticket-classification-system
```

---

### 2-> Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3-> Train the Model

```bash
python src/train_model.py
```

->>> This will:

* Load dataset from `data/sample_tickets.csv`
* Preprocess text
* Train the machine learning model
* Save model files in `models/` folder

---

### 4-> Run Prediction

```bash
python src/predict.py

--> run UI
streamlit run your_app.py

running the app form


