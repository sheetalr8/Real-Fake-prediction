import streamlit as st
import pandas as pd
import re
import string
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

# -----------------------------------
# Text Cleaning Function
# -----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# -----------------------------------
# Load or Train Model (Runs Once)
# -----------------------------------
@st.cache_resource
def load_or_train_model():

    # If model already saved → load
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer

    # Otherwise train model
    df = pd.read_csv("Fake_Real_Data(in).csv", engine="python")

    # IMPORTANT: Make sure these column names match your dataset
    # If column name is different, change here
    df['Text'] = df['Text'].apply(clean_text)

    X = df['Text']
    y = df['label']   # Ensure this column exists

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vectorized, y)

    # Save model & vectorizer
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


model, vectorizer = load_or_train_model()

# -----------------------------------
# User Input Section
# -----------------------------------
user_input = st.text_area("✍️ Enter News Text Here:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])

        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        # Correct probability mapping
        class_labels = model.classes_
        prob_dict = dict(zip(class_labels, probabilities))

        # Assuming:
        # 0 = Fake
        # 1 = Real
        fake_prob = round(prob_dict.get(0, 0) * 100, 2)
        real_prob = round(prob_dict.get(1, 0) * 100, 2)

        if prediction == 1:
            st.success("✅ This News is REAL")
        else:
            st.error("🚨 This News is FAKE")

        st.write(f"Fake Probability: {fake_prob}%")
        st.write(f"Real Probability: {real_prob}%")
