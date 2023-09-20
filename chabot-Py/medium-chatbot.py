import nltk
import random
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model

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
tokenizer = tf.keras.layers.TextVectorization(max_tokens = 1000, output_mode="int")
tokenizer.adapt(np.array(preguntas))

#modelamos con chatbot
model = tf.keras.Sequential([
    tokenizer,
    tf.keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=64, mask_zero=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(tokenizer.get_vocabulary()), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

#entrenamos el modelo
X = tokenizer(np.array(preguntas)).numpy()
Y = np.array(tokenizer(respuestas)).numpy()

model.fit(X, Y, epochs=50)

