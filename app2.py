from flask import Flask, request, jsonify, redirect, url_for, render_template
import openai
import faiss
import numpy as np
import os
import logging
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

# Załaduj zmienne środowiskowe z pliku .env (jeśli istnieje)
load_dotenv()

# Pobierz klucz API OpenAI z zmiennej środowiskowej
openai_api_key = os.getenv("OPENAI_API_KEY")

# Logowanie informacji o załadowaniu klucza API
logging.info(f"OPENAI_API_KEY loaded: {'Yes' if openai_api_key else 'No'}")

if not openai_api_key:
    raise ValueError("Brak klucza API OpenAI. Ustaw zmienną środowiskową OPENAI_API_KEY.")

# Ustawienie klucza API dla biblioteki OpenAI
openai.api_key = openai_api_key

app = Flask(__name__)

def get_embeddings(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

# Załaduj indeks FAISS
def load_faiss_index(index_path='faiss_index.idx'):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            logging.info(f"Indeks FAISS załadowany z {index_path}")
            return index
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return None
    else:
        logging.error(f"Plik {index_path} nie istnieje.")
        return None

# Załaduj dane źródłowe (fragmenty tekstu)
def load_data(file_path='dane.txt'):
    if not os.path.exists(file_path):
        logging.error(f"Plik {file_path} nie istnieje.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n\n')  # Dostosuj w zależności od sposobu chunkingu
    logging.info(f"Załadowano {len(data)} fragmentów danych z {file_path}")
    return data

# Załaduj indeks FAISS
faiss_index = load_faiss_index('faiss_index.idx')

if faiss_index is None:
    logging.error("Indeks FAISS nie został załadowany. Sprawdź, czy plik faiss_index.idx istnieje i jest poprawny.")
    # Opcjonalnie: Możesz zbudować indeks FAISS tutaj, jeśli nie istnieje

# Załaduj dane źródłowe
data_chunks = load_data('dane.txt')

# Logowanie liczby fragmentów i wektorów
logging.info(f"Liczba fragmentów danych: {len(data_chunks)}")
if faiss_index:
    logging.info(f"Liczba wektorów w indeksie FAISS: {faiss_index.ntotal}")

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
    query_embedding = get_embeddings(user_query)
    if not query_embedding:
        logging.error("Nie udało się wygenerować embeddingu.")
        return jsonify({'error': 'Nie udało się wygenerować embeddingu.'}), 500

    # Konwertuj embedding do formatu FAISS
    query_vector = np.array([query_embedding]).astype('float32')

    # Wyszukiwanie najbliższych sąsiadów
    k = 3  # Możesz dostosować wartość k
    try:
        D, I = faiss_index.search(query_vector, k)
        logging.info(f"Wyniki wyszukiwania FAISS - Dystanse: {D}, Indeksy: {I}")
    except Exception as e:
        logging.error(f"Error during FAISS search: {str(e)}")
        return jsonify({'error': 'Błąd podczas wyszukiwania w indeksie.'}), 500

    # Zbierz odpowiednie fragmenty
    relevant_chunks = [data_chunks[i] for i in I[0] if i >= 0 and i < len(data_chunks)]

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

        if not response.choices or not response.choices[0].message.content:
            logging.error("Brak zawartości w odpowiedzi od modelu.")
            return jsonify({'error': 'Brak odpowiedzi od modelu.'}), 500

        answer = response.choices[0].message.content.strip()
        logging.info(f"Wygenerowano odpowiedź: {answer}")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({'error': 'Nie udało się wygenerować odpowiedzi.'}), 500

    return jsonify({'answer': answer})

# Endpoint testowy
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
        if not response.choices or not response.choices[0].message.content:
            return jsonify({'error': 'Brak odpowiedzi od modelu.'}), 500
        answer = response.choices[0].message.content.strip()
        return f"<html><body><p>{answer}</p></body></html>"
    except Exception as e:
        logging.error(f"Error during test: {e}")
        return jsonify({'error': 'Błąd podczas testowania modelu.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
