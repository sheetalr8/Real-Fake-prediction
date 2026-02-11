import streamlit as st
import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# -----------------------------
# Load & Train Model (Cached)
# -----------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("Fake_Real_Data(in).csv")

    # Make sure column names are correct
    df.columns = ["text", "label"]

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].str.upper()

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vectorized = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vectorized, y)

    return model, vectorizer


model, vectorizer = train_model()

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

        # Map probabilities correctly
        prob_dict = dict(zip(model.classes_, probabilities))

        fake_prob = round(prob_dict.get("FAKE", 0) * 100, 2)
        real_prob = round(prob_dict.get("REAL", 0) * 100, 2)

        if prediction == "FAKE":
            st.error("🚨 This News is FAKE")
        else:
            st.success("✅ This News is REAL")

        st.write(f"Fake Probability: {fake_prob}%")
        st.write(f"Real Probability: {real_prob}%")
