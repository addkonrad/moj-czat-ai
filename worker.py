# worker.py

import os
import time
import dropbox
import logging
import openai
import pinecone
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import docx
import PyPDF2
import speech_recognition as sr
from pydub import AudioSegment
import threading

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Załaduj zmienne środowiskowe
load_dotenv()

# Pobierz klucze API
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Inicjalizacja klienta Dropbox
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Inicjalizacja klienta OpenAI
openai.api_key = OPENAI_API_KEY

# Inicjalizacja Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX)

def extract_text(file_path, file_type):
    text = ""
    try:
        if file_type == 'docx':
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif file_type == 'xlsx':
            df = pd.read_excel(file_path)
            text = df.to_string()
        elif file_type == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + '\n'
        elif file_type in ['mp3', 'wav']:
            audio = AudioSegment.from_file(file_path)
            audio.export("temp.wav", format="wav")
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp.wav") as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="pl-PL")
            os.remove("temp.wav")
        else:
            logging.warning(f"Unsupported file type: {file_type}")
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
    return text

def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def upload_to_pinecone(id, embedding, metadata):
    try:
        index.upsert(
            vectors=[
                {
                    "id": str(id),
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        logging.info(f"Uploaded embedding to Pinecone with ID: {id}")
    except Exception as e:
        logging.error(f"Error uploading to Pinecone: {e}")

def poll_dropbox():
    while True:
        try:
            # Lista plików w folderze Dropboxa
            folder_path = "/Addictive sound/AI Czat Bot"
            result = dbx.files_list_folder(folder_path)

            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    file_name = entry.name
                    file_path = entry.path_lower

                    # Pobierz plik
                    local_path = os.path.join("downloads", file_name)
                    dbx.files_download_to_file(local_path, file_path)

                    # Przetwarzanie pliku
                    file_ext = file_name.split('.')[-1].lower()
                    logging.info(f"Processing file: {file_name}")

                    text = extract_text(local_path, file_ext)
                    if not text:
                        logging.warning(f"No text extracted from {file_name}")
                        os.remove(local_path)
                        continue

                    response = generate_embedding(text)
                    if not response:
                        logging.error(f"Failed to generate embedding for {file_name}")
                        os.remove(local_path)
                        continue

                    embedding = response['data'][0]['embedding']

                    metadata = {
                        "filename": file_name,
                        "file_type": file_ext,
                        "text": text
                    }
                    upload_to_pinecone(file_name, embedding, metadata)

                    # Usuń plik lokalny po przetworzeniu
                    os.remove(local_path)

            # Poczekaj przed kolejnym sprawdzeniem
            time.sleep(60)  # Sprawdzaj co minutę

        except Exception as e:
            logging.error(f"Error polling Dropbox: {e}")
            time.sleep(60)  # Poczekaj przed ponownym sprawdzeniem

if __name__ == "__main__":
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
    poll_dropbox()
