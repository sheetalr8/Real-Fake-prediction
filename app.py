import streamlit as st
import pandas as pd
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# -----------------------------
# Load & Train Model Once
# -----------------------------
@st.cache_resource
def load_or_train_model():

    # If model already saved → load it
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer

    # Else train new model
    df = pd.read_csv("Fake_Real_Data(in).csv", engine="python")

    # IMPORTANT: Check your column names here
    # If your column name is 'Text' keep it.
    # If 'text', change accordingly.
    df['Text'] = df['Text'].apply(clean_text)

    X = df['Text']
    y = df['label']   # Make sure this column exists

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vectorized, y)

    # Save for future use
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


model, vectorizer = load_or_train_model()

# -----------------------------
# User Input Section
# -----------------------------
user_input = st.text_area("✍️ Enter News Text Here:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])

        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        # Map correct probabilities to classes
        class_labels = model.classes_
        prob_dict = dict(zip(class_labels, probabilities))

        # Adjust based on your label encoding
        # Assuming:
        # 0 = Fake
        # 1 = Real
        fake_prob = round(prob_dict.get(0, 0))
