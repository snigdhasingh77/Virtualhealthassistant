# 🩺 Virtual Health Assistant

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/model-BioBERT%20%7C%20MiniLM%20%7C%20ClinicalBERT-blueviolet)

A Flask-based NLP-powered assistant that delivers reliable, context-aware medical information. It intelligently combines a local database of verified Q&A with real-time external medical search powered by the Google Custom Search API. The system leverages domain-specific Transformer models like BioBERT and ClinicalBERT, alongside semantic sentence embeddings (MiniLM), to provide explainable and relevant responses.

---

## 📚 Table of Contents

- 🚀 Features
- 🧱 Tech Stack
- 🧠 Architecture
- 🔄 Example Query Flow
- 🌍 Project Motivation
- 🛠 How to Run Locally
- 📝 Related Blog
- 🪪 License

---

## 🚀 Features

-  Predicts whether a query is medically relevant using BioBERT and ClinicalBERT  
-  Searches a local MedQuAD-derived database for trusted answers  
-  Performs semantic similarity search via MiniLM embeddings  
-  Falls back to Google Custom Search API if no match is found  
-  Extracts, ranks, and parses live web content using BeautifulSoup  
-  Stores new medical queries + embeddings for future lookup (continuous learning)  
-  Secure configuration using dotenv (API keys + tokens)  

---

## 🧱 Tech Stack

| Layer             | Tools & Frameworks                                       |
|------------------|-----------------------------------------------------------|
| Frontend/API     | Flask, REST                                               |
| Database         | SQLite + SQLAlchemy ORM                                   |
| NLP Models       | BioBERT, ClinicalBERT, MiniLM (sentence-transformers)     |
| External APIs    | Google Custom Search                                      |
| Content Parsing  | BeautifulSoup                                             |
| Deployment Ready | dotenv, gunicorn-ready                                    |

---

## 🧠 Architecture

1. User submits a medical question via GET endpoint (/s?q=...)
2. Local DB queried for an exact match → return if found  
3. If no match, similarity is checked via sentence embeddings  
4. If still unmatched, classify query using BioBERT & ClinicalBERT  
5. If relevant → search Google, extract content, rank relevance  
6. Return content & store Q+A pair with embeddings for next time  

---

## 🔄 Example Query Flow

> Query: “How do I treat mild fever without antibiotics?”

- BioBERT: Medical intent → ✅  
- Local DB match: ❌  
- Semantic similarity: ❌  
- Triggers Google Search: ✅  
- Snippet scraped → cleaned → saved  
- Response returned to user + added to DB  

---

## 🌍 Project Motivation

In regions with limited healthcare access, this tool provides a fast, informative, and explainable bridge to medical knowledge — while continually improving with use. The system prioritizes:

- Verified local responses  
- Then semantically similar past queries  
- Then dynamic external results — all in real-time  

---

## 🛠 How to Run Locally

1. Clone the repository

2. Create a .env file with the following keys:

```env
GOOGLE_API_KEY=your_key_here  
CSE_ID=your_custom_search_id  
HUGGINGFACE_TOKEN=your_hf_token  
DATABASE_PATH=your_db_path (or leave default)
```

3. Install dependencies:

pip install -r requirements.txt

4. Run the app:

python app.py

→ Access via localhost:5000  

---

## 📝 Related Blog

👉 How I Built an NLP Health Assistant Using BioBERT, LangModels & Google Search

---

## 🪪 License

MIT License
