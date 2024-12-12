from flask import Flask, request, jsonify, render_template
import openai
import pinecone
import numpy as np
import os
import logging
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

# Załaduj zmienne środowiskowe
load_dotenv()

# Pobierz klucze API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Inicjalizacja klienta OpenAI
openai.api_key = OPENAI_API_KEY

# Inicjalizacja Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX)

app = Flask(__name__)

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query')

    logging.info(f"Otrzymano zapytanie: {user_query}")

    if not user_query:
        logging.error("Brak zapytania użytkownika.")
        return jsonify({'error': 'Brak zapytania użytkownika.'}), 400

    # Generowanie embeddingu dla zapytania użytkownika
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        logging.error("Nie udało się wygenerować embeddingu.")
        return jsonify({'error': 'Nie udało się wygenerować embeddingu.'}), 500

    # Konwertuj embedding do formatu Pinecone
    query_vector = np.array([query_embedding]).astype('float32')

    # Wyszukiwanie najbliższych sąsiadów
    k = 3  # Możesz dostosować wartość k
    try:
        results = index.query(query_vector.tolist(), top_k=k, include_metadata=True)
        logging.info(f"Wyniki wyszukiwania Pinecone: {results}")
    except Exception as e:
        logging.error(f"Error during Pinecone search: {str(e)}")
        return jsonify({'error': 'Błąd podczas wyszukiwania w Pinecone.'}), 500

    # Zbierz odpowiednie fragmenty
    relevant_chunks = [match.metadata['text'] for match in results['matches']]

    if not relevant_chunks:
        logging.warning("Nie znaleziono odpowiednich fragmentów. Odpowiadanie bez kontekstu.")
        prompt_text = f"Pytanie: {user_query}\nOdpowiedź:"
    else:
        # Tworzenie kontekstu dla modelu generatywnego
        context = "\n\n".join(relevant_chunks)
        prompt_text = f"{context}\n\nPytanie: {user_query}\nOdpowiedź:"

    logging.info(f"Tworzenie promptu dla modelu: {prompt_text[:100]}...")  # Logowanie pierwszych 100 znaków promptu

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Możesz dostosować model
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            max_tokens=2500  # Zwiększenie limitu tokenów
        )
        logging.info(f"Pełna odpowiedź OpenAI: {response}")

        if not response.choices or not response.choices[0].message['content']:
            logging.error("Brak zawartości w odpowiedzi od modelu.")
            return jsonify({'error': 'Brak odpowiedzi od modelu.'}), 500

        answer = response.choices[0].message['content'].strip()
        logging.info(f"Wygenerowano odpowiedź: {answer}")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({'error': 'Nie udało się wygenerować odpowiedzi.'}), 500

    return jsonify({'answer': answer})

@app.route('/test', methods=['GET'])
def test():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Możesz dostosować model
            messages=[
                {
                    "role": "user",
                    "content": "Cześć, jak się masz?"
                }
            ],
            max_tokens=150  # Możesz dostosować limit tokenów
        )
        if not response.choices or not response.choices[0].message['content']:
            return jsonify({'error': 'Brak odpowiedzi od modelu.'}), 500
        answer = response.choices[0].message['content'].strip()
        return f"<html><body><p>{answer}</p></body></html>"
    except Exception as e:
        logging.error(f"Error during test: {e}")
        return jsonify({'error': 'Błąd podczas testowania modelu.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
