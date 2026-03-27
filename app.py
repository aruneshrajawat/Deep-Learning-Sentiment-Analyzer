import streamlit as st
import re
import nltk
import torch
import numpy as np
import pickle
import torch.nn as nn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# -------------------------------
# NLTK Setup
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Sentiment Analyzer", layout="wide")

# -------------------------------
# CUSTOM CSS (🔥 PREMIUM UI)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.big-title {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #ccc;
}
.box {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL CLASS
# -------------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# -------------------------------
# LOAD MODEL
# -------------------------------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = RNN(5000)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    tokens = [word for word in tokens if word not in stop_words]

    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="big-title">🎬 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze movie reviews using Deep Learning</div>', unsafe_allow_html=True)

# -------------------------------
# SAMPLE BUTTON
# -------------------------------
if st.button("✨ Try Sample Review"):
    st.session_state.sample = "This movie was absolutely amazing and inspiring!"

user_input = st.text_area("Enter Review", value=st.session_state.get("sample", ""))

# -------------------------------
# HISTORY INIT
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# PREDICT
# -------------------------------
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = clean_text(user_input)

        vector = tfidf.transform([cleaned]).toarray()
        vector = torch.from_numpy(vector).float().unsqueeze(1)

        with st.spinner("Analyzing sentiment..."):
            with torch.no_grad():
                output = model(vector)
                prob = torch.sigmoid(output).item()

        # Save history
        st.session_state.history.append((user_input, prob))

        # -------------------------------
        # RESULT UI
        # -------------------------------
        st.markdown("### 📊 Prediction Result")

        st.progress(int(prob * 100))
        st.write(f"Confidence: **{prob*100:.2f}%**")

        if prob > 0.8:
            st.success("🔥 Highly Positive Review")
            st.balloons()
        elif prob > 0.6:
            st.success("🙂 Positive Review")
        elif prob > 0.4:
            st.warning("😐 Neutral Review")
        else:
            st.error("😡 Negative Review")

        # -------------------------------
        # CLEANED TEXT
        # -------------------------------
        with st.expander("🔍 See Processed Text"):
            st.write(cleaned)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 About Project")
st.sidebar.write("""
• Model: RNN (PyTorch)
• Features: TF-IDF (5000)
• Accuracy: ~87%
• Built by Arunesh Singh Rajawat
""")

# -------------------------------
# HISTORY
# -------------------------------
st.sidebar.title("🧠 History")
for text, p in st.session_state.history[-5:]:
    st.sidebar.write(f"→ {text[:25]}... ({p:.2f})")