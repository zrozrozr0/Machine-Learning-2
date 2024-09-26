import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Descargar los recursos de nltk necesarios
# nltk.download('stopwords')
# nltk.download('punkt')

# Función para cargar el léxico de emojis
def load_emoji_lexicon(filepath):
    emoji_df = pd.read_excel(filepath)
    emoji_dict = {}
    for _, row in emoji_df.iterrows():
        emoji_dict[row['emoji']] = row[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative', 'positive']].to_dict()
    return emoji_dict

# Función de preprocesamiento con manejo de negaciones y repeticiones
def preprocess_text_with_emojis_and_negations(text, emoji_dict):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    
    # Manejo de negaciones en español
    negations = {"no": "no", "nunca": "nunca", "nadie": "nadie", "ningún": "ningún",
                 "ninguna": "ninguna", "ninguno": "ninguno", "jamás": "jamás"}
    
    tokens = nltk.word_tokenize(text, language='english')
    for i in range(len(tokens) - 1):
        if tokens[i] in negations:
            tokens[i + 1] = "NO_" + tokens[i + 1]
    
    text = ' '.join(tokens)
    
    # Normalización de repeticiones de caracteres
    #text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # Eliminación de caracteres no alfabéticos
    text = re.sub(r'[^a-zA-ZñÑ\s]', '', text)
    tokens = nltk.word_tokenize(text, language='english')
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word for word in tokens if word not in stopwords]
    preprocessed_text = ' '.join(tokens)

    for emoji, sentiments in emoji_dict.items():
        if emoji in text:
            for sentiment, value in sentiments.items():
                if value > 0:
                    preprocessed_text += f' {sentiment}'

    return preprocessed_text

# Cargar el archivo xlsx del corpus y del léxico de emojis
df = pd.read_excel('Rest_Mex_2022.xlsx')
emoji_dict = load_emoji_lexicon('Emojis lexicon.xlsx')

# Asegurarse de que las columnas son de tipo string antes de concatenarlas
df['Title'] = df['Title'].astype(str)
df['Opinion'] = df['Opinion'].astype(str)

# Concatenar las columnas de título y opinión
df['text'] = df['Title'] + ' ' + df['Opinion']
df['text'] = df['text'].apply(lambda x: preprocess_text_with_emojis_and_negations(x, emoji_dict))

# Dividir en conjuntos de entrenamiento y prueba
X = df['text']
y = df['Polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, test_size=0.2, random_state=0)

# Definir los vectorizadores para las representaciones de texto
vectorizer_tfidf = TfidfVectorizer()

# Lista de modelos para entrenamiento y evaluación
models = [
    ('Logistic Regression', LogisticRegression(C=5.1,max_iter=1000, n_jobs=-1), vectorizer_tfidf)
]

results = []

# Entrenamiento y evaluación de cada modelo
for name, model, vectorizer in models:
    # Vectorizar el texto
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Balancear las clases en el conjunto de entrenamiento
    smote = SMOTE(random_state=0)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    
    # Entrenar el modelo
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predecir las etiquetas para el conjunto de prueba
    y_pred = model.predict(X_test_vec)
    
    # Calcular métricas de evaluación
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Almacenar los resultados
    results.append((name, vectorizer._class.name_, f1_macro))
    
    # Imprimir reporte de clasificación y matriz de confusión para cada modelo
    print(f'Model: {name} with {vectorizer._class.name_}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f'F1 Macro Score: {f1_macro}\n')

# Mostrar los resultados
results_df = pd.DataFrame(results, columns=['Model', 'Vectorizer', 'F1 Macro Score'])
print(results_df)

# Seleccionar el mejor resultado
best_result = results_df.loc[results_df['F1 Macro Score'].idxmax()]
print(f'Best Model: {best_result["Model"]} with {best_result["Vectorizer"]}, F1 Macro Score: {best_result["F1 Macro Score"]}')