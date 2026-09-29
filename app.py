import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

# Simple preprocessing without NLTK
def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load models
model_lr = joblib.load("model_logistic_regression.pkl")
model_rf = joblib.load("model_random_forest.pkl")
model_nb = joblib.load("model_naive_bayes.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.title("🔍 Fake News Detector")

st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    ["Logistic Regression", "Random Forest", "Naive Bayes"]
)

model_map = {
    "Logistic Regression": model_lr,
    "Random Forest": model_rf,
    "Naive Bayes": model_nb
}

selected_model_obj = model_map[selected_model]

st.markdown("---")

st.subheader("📝 Enter News Article")
news_text = st.text_area("Paste your news article:", height=200)

if st.button("🔍 Predict", type="primary"):
    if news_text.strip():
        processed_text = simple_preprocess(news_text)
        
        if processed_text:
            X = tfidf.transform([processed_text])
            prediction = selected_model_obj.predict(X)[0]
            probabilities = selected_model_obj.predict_proba(X)[0]
            confidence = probabilities[int(prediction)]
            
            prob_fake = probabilities[0]
            prob_real = probabilities[1]
            
            st.markdown("---")
            
            if prediction == 0:
                st.markdown("""
                <div style="background-color: #ff6b6b; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h2>🔴 FAKE NEWS</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #51cf66; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h2>🟢 REAL NEWS</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Confidence", f"{confidence*100:.2f}%")
            with col2:
                st.metric("🔴 Fake Prob", f"{prob_fake*100:.2f}%")
            with col3:
                st.metric("🟢 Real Prob", f"{prob_real*100:.2f}%")
            
            st.markdown("---")
            st.subheader("📊 Probability Chart")
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            classes = ['FAKE', 'REAL']
            probs = [prob_fake, prob_real]
            colors = ['#ff6b6b', '#51cf66']
            bars = ax.bar(classes, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("📋 Article Stats")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Characters", len(news_text))
            with col_s2:
                st.metric("Words", len(news_text.split()))
            with col_s3:
                st.metric("Sentences", len(news_text.split('.')))
        else:
            st.error("❌ Could not process text!")
    else:
        st.error("❌ Please enter text!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    🔍 Fake News Detection System v1.0 | Always verify from credible sources
</div>
""", unsafe_allow_html=True)