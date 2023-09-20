import nltk
import string
import random
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('wordnet')

data = [
    {"pregunta": "Hola", "respuesta": "¡Hola! ¿En qué puedo ayudarte?"},
    {"pregunta": "¿Cómo estás?", "respuesta": "Estoy bien, gracias."},
    {"pregunta": "Adiós", "respuesta": "Hasta luego. ¡Que tengas un buen día!"}
    #Se agregan más preguntas
]

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

#preprocesamos los datos
preprocessed_data = [(preprocess(item["pregunta"]), preprocess(item["respuesta"])) for item in data]

#separamos las preguntas y las respuestas
preguntas, respuestas = zip(*preprocessed_data)

#tokeniza las preguntas y respuestas
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preguntas + respuestas)
preguntas_tokenizadas = tokenizer.texts_to_sequences(preguntas)
respuestas_tokenizadas = tokenizer.texts_to_sequences(respuestas)

#añadimos paddin
max_sequence_length = max(max(len(seq) for seq in preguntas_tokenizadas), max(len(seq) for seq in respuestas_tokenizadas))
preguntas_tokenizadas = pad_sequences(preguntas_tokenizadas, maxlen=max_sequence_length, padding='post')
respuestas_tokenizadas = pad_sequences(respuestas_tokenizadas, maxlen=max_sequence_length, padding='post')

#modelamos con chatbot
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, mask_zero=True),
    LSTM(128),
    Dense(len(tokenizer.word_index) + 1, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

#entrenamos el modelo
model.fit(preguntas_tokenizadas, respuestas_tokenizadas, epochs=50)

def obtener_respuesta(pregunta):
    pregunta_preprocesada = preprocess(pregunta)
    pregunta_tokenizada = tokenizer.texts_to_sequences([pregunta_preprocesada])
    respuesta_tokenizada = model.predict(pregunta_tokenizada)
    respuesta_tokenizada = np.argmax(respuesta_tokenizada, axis=-1)
    respuesta_texto = tokenizer.sequences_to_texts(respuesta_tokenizada)
    return respuesta_texto[0]


print("¡Hola! Soy un chatbot. Puedes escribir 'Adiós' en cualquier momento para salir.")
while True:
    usuario_input = input("Tu: ")
    if usuario_input.lower() == "adiós":
        print("Chatbot: Hasta luego. ¡Que tengas un buen día!")
        break
    else:
        respuesta = obtener_respuesta(usuario_input)
        print("Chatbot: ", respuesta)