from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import requests
import html
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DATABASE_PATH = os.getenv('DATABASE_PATH', 'D:/Pythonproject/Virtualhealthassistant/data/medquad.db')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CSE_ID = os.getenv('CSE_ID')
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
CLINICALBERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_AUTH_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class QuestionAnswer(Base):
    __tablename__ = 'medquad'
    question = Column('Question', String, primary_key=True)
    answer = Column('answer', String)
    focus_area = Column('focus area', String)
    embedding = Column('embedding', String)

Base.metadata.create_all(engine)

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    return tokenizer, model

biobert_tokenizer, biobert_model = load_model_and_tokenizer(BIOBERT_MODEL_NAME)
clinicalbert_tokenizer, clinicalbert_model = load_model_and_tokenizer(CLINICALBERT_MODEL_NAME)

def load_embedding_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    model = AutoModel.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    return tokenizer, model

embedding_tokenizer, embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

def predict_with_model(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions

def get_embeddings(text):
    inputs = embedding_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def is_medical_related(text):
    try:
        biobert_predictions = predict_with_model(biobert_tokenizer, biobert_model, text)
        biobert_score = biobert_predictions[0][1].item()

        clinicalbert_predictions = predict_with_model(clinicalbert_tokenizer, clinicalbert_model, text)
        clinicalbert_score = clinicalbert_predictions[0][1].item()
        
        return biobert_score > 0.55 or clinicalbert_score > 0.55
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return False

def search_db(query):
    try:
        result = session.query(QuestionAnswer).filter(QuestionAnswer.question.ilike(f'%{query}%')).first()
        if result:
            return result.answer
        return None
    except Exception as e:
        print(f"Error querying database: {e}")
        return None

def search_google(query):
    try:
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {'key': GOOGLE_API_KEY, 'cx': CSE_ID, 'q': query}
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_results = response.json()

        if 'items' in search_results and len(search_results['items']) > 0:
            results = []
            for item in search_results['items']:
                snippet = item.get('snippet', 'No snippet available')
                snippet = html.unescape(snippet)
                link = item.get('link', 'No link available')

                try:
                    content_response = requests.get(link)
                    content_response.raise_for_status()
                    soup = BeautifulSoup(content_response.content, 'html.parser')
                    content = soup.get_text()
                except Exception as e:
                    content = "Content extraction failed"

                result = {
                    'title': item.get('title', 'No title available'),
                    'snippet': snippet,
                    'link': link,
                    'formatted_url': item.get('displayLink', 'No formatted URL available'),
                    'full_content': content
                }
                results.append(result)

            relevant_results = [res for res in results if 'medical' in res['full_content'].lower()]
            if relevant_results:
                return {'results': relevant_results, 'source': 'google'}
        return {'result': 'No relevant information found', 'source': 'google'}
    except requests.RequestException as e:
        print(f"Google API error: {e}")
        return {'result': 'Error fetching results', 'source': 'none'}

def update_db(question, answer):
    try:
        embedding = get_embeddings(question).flatten().tolist()
        embedding_str = ','.join(map(str, embedding))
        new_entry = QuestionAnswer(question=question, answer=answer, focus_area=None, embedding=embedding_str)
        session.add(new_entry)
        session.commit()
    except Exception as e:
        print(f"Error updating database: {e}")

def find_similar_query(query, threshold=0.8):
    try:
        query_embedding = get_embeddings(query)
        all_entries = session.query(QuestionAnswer).all()
        
        for entry in all_entries:
            if entry.embedding:
                stored_embedding = np.array(list(map(float, entry.embedding.split(','))))
                similarity = cosine_similarity(query_embedding, stored_embedding.reshape(1, -1))
                
                if similarity >= threshold:
                    return entry.answer
    except Exception as e:
        print(f"Error in similarity search: {e}")
    return None

@app.route('/s', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Check if the query exists in the database
    result = search_db(query)
    if result:
        return jsonify({"result": result, "source": "database"}), 200

    # Find similar queries
    similar_result = find_similar_query(query)
    if similar_result:
        return jsonify({"result": similar_result, "source": "database"}), 200

    # Determine if the query is medical-related
    is_medical = is_medical_related(query)
    
    if is_medical:
        result = search_google(query)
        if result and 'results' in result and len(result['results']) > 0:
            relevant_content = result['results'][0]['full_content']
            
            existing_entry = session.query(QuestionAnswer).filter(QuestionAnswer.question == query).first()
            if existing_entry:
                existing_entry.answer = relevant_content
                existing_entry.embedding = ','.join(map(str, get_embeddings(query).flatten().tolist()))
                session.commit()
            else:
                update_db(query, relevant_content)
            
            return jsonify({"result": relevant_content, "source": "google"}), 200
    else:
        return jsonify({"result": "No relevant information found.", "source": "none"}), 404

if __name__ == '__main__':
    app.run(debug=True)


