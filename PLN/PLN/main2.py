import re
import spacy
# Leer el archivo "corpus_noticias.txt"
with open("corpus_noticias.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

# Función de tokenización simple usando expresiones regulares
def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Tokenizar y lematizar el contenido extraído
def lemmatize(tokens):
    lemmas = {
        "news": "noticia",
        "car": "coche",

    }
    lemmatized_tokens = [lemmas.get(word, word) for word in tokens]
    return lemmatized_tokens


stopwords = [
    "a", "al", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "e", "el",
    "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres",
    "es", "esa", "esas", "ese", "eses", "esto", "estos", "fue", "fueron", "fuiste",
    "fuimos", "ha", "han", "has", "hasta", "hay", "l", "la", "las", "le", "les", "lo",
    "los", "me", "mi", "mis", "mucho", "muchos", "nada", "ni", "no", "nos", "nosotros",
    "o", "os", "para", "pero", "por", "porque", "que", "quien", "se", "sean", "si",
    "sido", "sin", "sobre", "sois", "somos", "su", "sus", "también", "te", "tengo",
    "ti", "tu", "tus", "un", "una", "uno", "unos", "vosotros", "y", "ya", "del",
]

# También puedes agregar más palabras si es necesario


# Extraer la sección de noticias
noticias = re.split(r'&&&&&&&&[A-Za-z\s]+!', corpus)
corpus_noticias = [noticia.strip() for noticia in noticias if noticia.strip()]


for noticia in corpus_noticias:
    tokens = tokenize(noticia)
    lemmatized_tokens = lemmatize(tokens)
    filtered_tokens = [word for word in lemmatized_tokens if word not in stopwords]
    # Puedes hacer lo que desees con los tokens procesados aquí, como imprimirlos o guardarlos.
with open('corpus_noticias_limpio.txt', "w", encoding="utf-8") as f:
    f.writelines(filtered_tokens)
f.close()

print("Se ha creado el corpus normalizado")