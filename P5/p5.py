import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from bs4 import BeautifulSoup
from tabulate import tabulate

"""# Normalización del texto
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')"""

# Cargar el archivo CSV
df = pd.read_csv('raw data corpus.csv')
print("Corpus cargado con exito")

# Reemplazar valores nulos con cadenas vacías
df['Titulo'] = df['Titulo'].fillna('')
df['Resumen de Contenido'] = df['Resumen de Contenido'].fillna('')

# Concatenar 'titulo' y 'resumen contenido' en una nueva columna 'text'
df['text'] = df['Titulo'] + ' ' + df['Resumen de Contenido']

# Seleccionar características (X) y target (y)
X = df['text']
y = df['Seccion']

# División de los datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

# Configuración de stopwords y lematizador para procesamiento de texto
stop_words = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def text_cleaning(text):
    # Eliminar etiquetas HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Eliminar caracteres especiales y puntuación
    text = re.sub(r'[^\w\s]', '', text)
    return text

def normalize_text(text):
    # Convertir a minúsculas, eliminar caracteres especiales y números, tokenizar, eliminar stopwords y lematizar
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("Preparando la tabla")

# Definir los vectorizadores para las representaciones de texto
vectorizer_bin = CountVectorizer(binary=True)
vectorizer_freq = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()

# Entrenamiento y evaluación de modelos
models = [
    ('Logistic Regression', LogisticRegression(max_iter=200), 'Tokenization + stopwords + lemmatization', 'binarized', vectorizer_bin),
    ('Logistic Regression', LogisticRegression(max_iter=200), 'Tokenization + stopwords + lemmatization', 'frequency', vectorizer_freq),
    ('Logistic Regression', LogisticRegression(max_iter=200), 'Tokenization + stopwords + lemmatization', 'tf-idf', vectorizer_tfidf),
    ('Naive Bayes', MultinomialNB(), 'Tokenization + stopwords + lemmatization', 'binarized', vectorizer_bin),
    ('Naive Bayes', MultinomialNB(), 'Tokenization + stopwords + lemmatization', 'frequency', vectorizer_freq),
    ('Naive Bayes', MultinomialNB(), 'Tokenization + stopwords + lemmatization', 'tf-idf', vectorizer_tfidf),
    ('SVM', SVC(), 'Tokenization + stopwords + lemmatization', 'binarized', vectorizer_bin),
    ('SVM', SVC(), 'Tokenization + stopwords + lemmatization', 'frequency', vectorizer_freq),
    ('SVM', SVC(), 'Tokenization + stopwords + lemmatization', 'tf-idf', vectorizer_tfidf),
    ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(200, 100)), 'Tokenization + text_cleaning + stopwords + lemmatization', 'binarized', vectorizer_bin),
    ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(200, 100)), 'Tokenization + text_cleaning + stopwords + lemmatization', 'frequency', vectorizer_freq),
    ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(200, 100)), 'Tokenization + text_cleaning + stopwords + lemmatization', 'tf-idf', vectorizer_tfidf)
]

results = []

# Entrenamiento y evaluación de cada modelo
for name, model, norm_text, rep_name, vectorizer in models:
    # Limpiar y normalizar texto según el tipo de modelo
    if name == 'Multilayer Perceptron':
        X_train_rep = X_train.apply(text_cleaning)
        X_test_rep = X_test.apply(text_cleaning)
    else:
        X_train_rep = X_train.apply(normalize_text)
        X_test_rep = X_test.apply(normalize_text)
    
    # Vectorizar el texto
    X_train_rep = vectorizer.fit_transform(X_train_rep)
    X_test_rep = vectorizer.transform(X_test_rep)
    
    # Entrenar el modelo
    model.fit(X_train_rep, y_train)
    
    # Predecir las etiquetas para el conjunto de prueba
    y_pred = model.predict(X_test_rep)
    
    # Calcular métricas de evaluación
    f1_weighted = f1_score(y_test, y_pred, average='weighted') # Calcular el F1-score ponderado
    f1_macro = f1_score(y_test, y_pred, average='macro') # Calcular el F1-score macro
    accuracy = accuracy_score(y_test, y_pred) # Calcular la precisión
    
    # Obtener los parámetros específicos del modelo
    if name == 'Logistic Regression':
        params = {'max_iter': 200}
    elif name == 'Naive Bayes':
        params = {'default': 'parameters'}
    elif name == 'SVM':
        params = {'Kernel': 'Linear'}
    elif name == 'Multilayer Perceptron':
        params = {'hidden_layer_sizes': '(200, 100)'}
    
    # Almacenar los resultados en la lista
    results.append((name, params, norm_text, rep_name, f1_weighted, f1_macro, accuracy))

# Crear un DataFrame de resultados
headers = ['Machine learning method', 'ML method parameters', 'Text normalization', 'Text representation', 'Weighted F1-score', 'Macro F1-score', 'Accuracy']
results_df = pd.DataFrame(results, columns=headers)

# Imprimir la tabla en la consola (opcional)
# print(tabulate(results_df, headers=headers, tablefmt='grid'))

# Guardar la tabla en un archivo CSV
results_df.to_csv('results.csv', index=False)
print("Resultados guardados en 'results.csv'")
