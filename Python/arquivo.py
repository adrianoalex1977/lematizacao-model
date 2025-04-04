
# precisa instalar o dicionario em português
#python -m spacy download pt_core_news_lg

import spacy
import json

# Carregar o modelo de NLP em português
nlp = spacy.load("pt_core_news_lg")

# Exemplo de texto bíblico
texto = "Os discípulos estavam caminhando e falando sobre as escrituras."

# Processar o texto
doc = nlp(texto)

# Criar dicionário de lemas
lematizacao = {token.text: token.lemma_ for token in doc if not token.is_punct}

# Salvar em um arquivo JSON
with open("lematizacao.json", "w", encoding="utf-8") as arquivo:
    json.dump(lematizacao, arquivo, ensure_ascii=False, indent=4)

print("✅ Arquivo 'lematizacao.json' salvo com sucesso!")
