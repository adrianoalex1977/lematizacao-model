import spacy
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# Carregar o modelo de NLP em português
nlp = spacy.load("pt_core_news_lg")

# Exemplo de texto bíblico
texto = "Os discípulos estavam caminhando e falando sobre as escrituras."
doc = nlp(texto)

# Criar listas de palavras e lemas
palavras = [token.text.lower() for token in doc if not token.is_punct]
lemas = [token.lemma_.lower() for token in doc if not token.is_punct]

# Tokenizar palavras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(palavras)
sequencias = tokenizer.texts_to_sequences(palavras)
sequencias = np.array(sequencias).reshape(-1)

# Tokenizar lemas
lema_tokenizer = Tokenizer()
lema_tokenizer.fit_on_texts(lemas)
lemas_sequencias = lema_tokenizer.texts_to_sequences(lemas)
lemas_sequencias = np.array(lemas_sequencias).reshape(-1)

# Criar o modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(lema_tokenizer.word_index) + 1, activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo (exemplo didático, precisa de mais dados na prática)
modelo.fit(sequencias, lemas_sequencias, epochs=100, verbose=1)

# Salvar o modelo
modelo.save("modelo_lematizacao")

# Salvar os mapeamentos para conversão de volta para palavras
import json
with open("tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)

with open("lema_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(lema_tokenizer.word_index, f, ensure_ascii=False, indent=4)

print("✅ Modelo treinado e salvo com sucesso!")

