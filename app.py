import streamlit as st
import joblib
from PIL import Image

# Load model and vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

# Custom CSS for Styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .main-container {
        background: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #1e3a8a;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
    }
    .stButton>button {
        background: #1e3a8a;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: #1e40af;
    }
    </style>
""", unsafe_allow_html=True)

# App Layout
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.write("🔍 **Enter a News Article below to check whether it's Fake or Real.**")

# Input Box
input_text = st.text_area("✍️ Paste your news article here:", "")

# Check News Button
if st.button("🔎 Analyze News"):
    if input_text.strip():
        transformed_input = vectorizer.transform([input_text])
        prediction = model.predict(transformed_input)

        # Display result with emoji and styling
        if prediction[0] == 1:
            st.success("✅ The News is **Real!**")
        else:
            st.error("🚨 The News is **Fake!**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.write("📌 Developed by **Maryam Ayoub** | Powered by **Machine Learning**")

