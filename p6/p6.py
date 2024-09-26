import pandas as pd
import nltk
import re
import spacy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Asegúrate de que los recursos necesarios de NLTK estén descargados
# nltk.download('stopwords')
# nltk.download('punkt')

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_md")

# Función para corregir el texto automáticamente
def correct_text(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

# Función para preprocesar el texto
def preprocess_text(text):
    # Corrección automática del texto
    text = correct_text(text)
    
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar etiquetas HTML
    text = re.sub(r'<.*?>', '', text)
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    # Tokenización y lematización
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return ' '.join(tokens)

# Mejor manejo de la negación
def handle_negation(text):
    doc = nlp(text)
    tokens = []
    negation = False
    for token in doc:
        if token.dep_ == 'neg':
            negation = True
        if negation:
            tokens.append("no_" + token.lemma_)
            negation = False
        else:
            tokens.append(token.lemma_)
    return ' '.join(tokens)

# Cargar el archivo Excel
df = pd.read_excel('Rest_Mex_2022.xlsx')

# Asegúrate de que las columnas sean cadenas antes de concatenar
df['Title'] = df['Title'].astype(str)
df['Opinion'] = df['Opinion'].astype(str)

# Concatenar las columnas de título y opinión
df['text'] = df['Title'] + ' ' + df['Opinion']
df['text'] = df['text'].apply(preprocess_text).apply(handle_negation)

# Dividir en conjuntos de entrenamiento y prueba
X = df['text']
y = df['Polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Definir vectorizadores
vectorizer_bin = CountVectorizer(binary=True)
vectorizer_freq = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()

# Lista de modelos para entrenamiento y evaluación
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000), vectorizer_bin),
    ('Logistic Regression', LogisticRegression(max_iter=1000), vectorizer_freq),
    ('Logistic Regression', LogisticRegression(max_iter=1000), vectorizer_tfidf),
    # Descomenta estas líneas para agregar más modelos
    # ('Naive Bayes', MultinomialNB(), vectorizer_bin),
    # ('Naive Bayes', MultinomialNB(), vectorizer_freq),
    # ('Naive Bayes', MultinomialNB(), vectorizer_tfidf),
    # ('SVM', SVC(kernel='sigmoid'), vectorizer_bin),
    # ('SVM', SVC(kernel='sigmoid'), vectorizer_freq),
    # ('SVM', SVC(kernel='sigmoid'), vectorizer_tfidf),
    # ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(300, 100)), vectorizer_bin),
    # ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(300, 100)), vectorizer_freq),
    # ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(300, 100)), vectorizer_tfidf)
]

results = []

# Entrenar y evaluar cada modelo
for name, model, vectorizer in models:
    # Vectorizar el texto
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Balancear clases en el conjunto de entrenamiento
    smote = SMOTE(random_state=0)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    
    # Entrenar el modelo
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test_vec)
    
    # Calcular métricas de evaluación
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Almacenar resultados
    results.append((name, vectorizer.__class__.__name__, f1_macro))
    
    # Imprimir informe de clasificación y matriz de confusión
    print(f'Model: {name} with {vectorizer.__class__.__name__}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f'F1 Macro Score: {f1_macro}\n')

# Mostrar resultados
results_df = pd.DataFrame(results, columns=['Model', 'Vectorizer', 'F1 Macro Score'])
print(results_df)

# Seleccionar el mejor resultado
best_result = results_df.loc[results_df['F1 Macro Score'].idxmax()]
print(f'Best Model: {best_result["Model"]} with {best_result["Vectorizer"]}, F1 Macro Score: {best_result["F1 Macro Score"]}')
