import pandas as pd
import re
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import csv


# Leer el archivo CSV en un dataframe con el separador "&&&&&&&&"
df = pd.read_csv('corpus_noticias.txt', sep='&&&&&&&&', header=None, engine='python')

# Eliminar todas las columnas excepto la de noticias
df = df.drop(df.columns[[0, 1,3]], axis=1)

# Guardar la tercera columna en un archivo CSV
df.to_csv('seccionNoticias.csv', index=False, header = None)

nlp = spacy.load('es_core_news_sm')
nlp.max_length = 14880190

stop_words = set(STOP_WORDS)
stop_words.update(
    ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si', 'mientras', 'como', 'este', 'ese',
     'aquel', 'este', 'esta', 'ese', 'esa', 'aquel', 'aquella', 'es', 'soy', 'eres', 'es', 'somos', 'son', 'fui',
     'fuiste', 'fue', 'fuimos', 'fueron','yo', 'tu', 'el', 'ella', 'ello', 'nosotros',
     'vosotros', 'ellos', 'ellas', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'mío', 'tuyo', 'suyo', 'nuestro', 'vuestro',
     'mi', 'tu', 'su', 'nuestro', 'vuestro', 'me', 'te', 'se', 'nos', 'os', 'se'])

stop_words = spacy.lang.es.stop_words.STOP_WORDS

categorias_eliminar = ["DET", "ADP", "CONJ", "PRON"]

corpus = open('seccionNoticias.csv', 'r', encoding='utf-8').read()


doc = nlp(corpus)

lista_auxiliar = []
lista_listas =[]


for j, token in enumerate(doc):
    token_text_lower = token.text.lower()
    
    if token.is_space and '\n' in token.text:
        lista_auxiliar = [token.lemma_] + lista_auxiliar[1:-1]
        lista_listas.append(lista_auxiliar)
        lista_auxiliar = []

    elif token.is_stop and token.pos_ in categorias_eliminar:
        stop_words.discard(token.text)

    elif token_text_lower not in stop_words and not token.is_stop and '\n' not in token.text:
        lista_auxiliar.append(token.lemma_)

corpusNormalizado = open('corpusNormalizado.txt', 'w', encoding="utf-8")

for lines in lista_listas:
    for l in lines:
        if l != '\n':
            l += " "
        corpusNormalizado.writelines(l)