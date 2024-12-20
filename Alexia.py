import os
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr

# Configuration du chemin pour le fichier texte
file_path = r"questions_reponses_data_science.txt"

# Téléchargement des stopwords français
nltk.download('stopwords')
stopwords_fr = set(stopwords.words('french'))

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Prétraitement du texte avec spaCy
def preprocess(text):
    doc = nlp(text)
    words = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stopwords_fr]
    return " ".join(words)

# Fonction pour charger et prétraiter le fichier texte
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    questions, answers, processed_questions = [], [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
                questions.append(question)
                processed_questions.append(preprocess(question))
            elif line.startswith("Réponse:"):
                answer = line.replace("Réponse:", "").strip()
                answers.append(answer)
    return questions, processed_questions, answers

# Fonction pour trouver la réponse la plus pertinente
def find_best_response(user_query, questions, processed_questions, answers):
    user_query_processed = preprocess(user_query)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    query_tfidf = vectorizer.transform([user_query_processed])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()

    if similarities[best_match_index] > 0:
        return answers[best_match_index]
    return "Je ne sais pas répondre à cette question."

# Fonction pour enregistrer l'audio
def record_audio(duration=5, samplerate=44100, filename="output.wav"):
    st.info("Enregistrement audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Attendre la fin de l'enregistrement
    write(filename, samplerate, audio)  # Sauvegarder l'audio dans un fichier WAV
    st.success("Enregistrement terminé.")
    return filename

# Fonction pour convertir la parole en texte
def speech_to_text():
    filename = record_audio()
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)  # Lire le fichier audio
    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "Désolé, je n'ai pas compris ce que vous avez dit."
    except sr.RequestError as e:
        return f"Erreur avec le service de reconnaissance vocale : {e}"

# Interface utilisateur Streamlit
def chatbot():
    st.title("Chatbot Data Science")
    st.write("Posez une question via texte ou parole et obtenez une réponse pertinente !")

    try:
        questions, processed_questions, answers = load_data(file_path)
        st.success("Fichier chargé avec succès ! Posez votre question ci-dessous.")
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")
        return

    # Interface pour poser des questions
    input_mode = st.radio("Choisissez le mode d'entrée :", ("Texte", "Parole"))

    if input_mode == "Texte":
        user_query = st.text_input("Votre question :")
        if user_query:
            response = find_best_response(user_query, questions, processed_questions, answers)
            st.write(f"**Réponse :** {response}")

    elif input_mode == "Parole":
        if st.button("Cliquez pour parler"):
            user_query = speech_to_text()
            if user_query:
                st.write(f"**Vous avez dit :** {user_query}")
                response = find_best_response(user_query, questions, processed_questions, answers)
                st.write(f"**Réponse :** {response}")

if __name__ == "__main__":
    chatbot()
