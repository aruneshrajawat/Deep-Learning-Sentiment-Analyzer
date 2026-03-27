🎬 IMDB Sentiment Analysis (Deep Learning)

🚀 A powerful AI-based Sentiment Analysis Web App built using RNN + TF-IDF + Streamlit that predicts whether a movie review is Positive or Negative with confidence score.

---

🔥 Live Demo

👉 

---


🧠 Features

• ✅ Deep Learning model using Recurrent Neural Network (RNN)
• ✅ Text preprocessing (cleaning, stopwords removal, stemming)
• ✅ TF-IDF vectorization
• ✅ Real-time prediction using Streamlit UI
• ✅ Confidence score display
• ✅ Clean and modern UI

---

🏗️ Tech Stack

• Frontend: Streamlit
• Backend: Python
• Model: PyTorch (RNN)
• NLP: NLTK
• Vectorization: TF-IDF (Scikit-learn)

---

📂 Project Structure

IMDB-Sentiment-Analysis/
│
├── app.py
├── model.pth
├── tfidf.pkl
├── requirements.txt
├── notebooks/
│   └── RNN.ipynb
│
└── README.md

---

⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/yourusername/IMDB-Sentiment-Analysis-DeepLearning.git
cd IMDB-Sentiment-Analysis-DeepLearning

---

2️⃣ Install dependencies

pip install -r requirements.txt

---

3️⃣ Run the app

streamlit run app.py

---

🎯 How It Works

1. User enters a movie review
2. Text is cleaned (lowercase, remove URLs, punctuation, etc.)
3. Stopwords removed + stemming applied
4. Converted to numerical form using TF-IDF
5. Passed into trained RNN model
6. Output → Positive / Negative with probability

---

📊 Example

Input| Output
"I love this movie"| Positive 😊
"Worst film ever"| Negative 😡

---

🚀 Future Improvements

• 🔹 Add LSTM / GRU models
• 🔹 Deploy on cloud (AWS / Streamlit Cloud)
• 🔹 Add multi-language support
• 🔹 Improve UI with animations

---

👨‍💻 Author

Arunesh Singh Rajawat
🎓 B.Tech CSE | AI/ML Developer

---

⭐ Show Your Support

If you like this project, please ⭐ star the repository!

---

📜 License

This project is licensed under the MIT License.
