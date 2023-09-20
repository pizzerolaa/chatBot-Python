import random

#Diccionario de respuesta predefinidas
respuestas = {
    "hola": ["¿Qué tal?", "Hola, ¿en que puedo serte últil?", "¡Hola!"], 
    "¿como estas?": ["Estoy bien, gracias.", "Me siento de maravila", "Todo va genial"],
    "adios": ["Hasta pronto.", "Adiós, ten un buen día.", "Nos vemos luego!"],
    "nombre": ["Me llamo TilinBot.", "Puedes llamarme TilinBot.", "Mi nombre por el momento es TilinBot."],
    "default": ["No entiendo tu pregunta.", "Lo siento, mi capacidad aún es límitada.", "Por favor, reformula tu pregunta."],
}

def chatbot_resp(pregunta):
    pregunta = pregunta.lower() #convertimos pregunta a minuscula
    respuesta = random.choice(respuestas["default"])
    #buscamos una respuesta en el diccionario
    if pregunta in respuestas:
        respuesta = random.choice(respuestas[pregunta])
        
    return respuesta

print("Hola, soy TilinBot. Puedes escribir 'adios' en cualquier momento para salir.")

while True:
    usuario_input = input("Tu: ")
    if usuario_input.lower() == "adios":
            print("TilinBot: Hasta luego!!")
            break
    else:
        respuesta = chatbot_resp(usuario_input)
        print("TilinBot: ", respuesta)