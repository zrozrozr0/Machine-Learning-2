import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


# Función para cargar el nuevo documento
def cargar_documento():
    archivo_path = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    if archivo_path:
        with open(archivo_path, "r", encoding="utf-8") as archivo:
            new_document = archivo.read()
            new_document = limpiar_texto(new_document)  # Limpia el texto
            new_document_entry.delete(1.0, tk.END)
            new_document_entry.insert(tk.END, new_document)
            archivo_prueba_entry.delete(0, tk.END)
            archivo_prueba_entry.insert(0, archivo_path)


# Función para calcular la similitud y mostrar resultados (TF-IDF)
def calcular_similitudtfidf(news_corpus):
    archivo_prueba_path = archivo_prueba_entry.get()

    if not archivo_prueba_path:
        resultados_text.config(state=tk.NORMAL)
        resultados_text.delete(1.0, tk.END)
        resultados_text.insert(tk.END, "Por favor, seleccione un archivo de prueba primero.")
        resultados_text.config(state=tk.DISABLED)
        return

    with open(archivo_prueba_path, "r", encoding="utf-8") as archivo:
        new_document = archivo.read()
        new_document = limpiar_texto(new_document)  # Limpia el texto

    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(news_corpus)

    new_document_vector = vectorizer_tfidf.transform([new_document])

    similarities = cosine_similarity(new_document_vector, X_tfidf)

    most_similar_indices = similarities.argsort()[0][::-1][:10]

    # Muestra las noticias similares en el área de texto
    resultados_text.config(state=tk.NORMAL)
    resultados_text.delete(1.0, tk.END)
    for idx, similitud in zip(most_similar_indices, similarities[0][most_similar_indices]):
        noticia_similar = news_corpus[idx]
        codigo = noticia_similar.split('&&&&&&&&')[0]
        linea_codigo = news_corpus.index(noticia_similar) + 1  # Número de línea de código
        resultados_text.insert(tk.END, f"Similitud TF-IDF: {similitud:.2f} - Línea {linea_codigo}\n")
    resultados_text.config(state=tk.DISABLED)


# Función para calcular la similitud y mostrar resultados (Binarizado)
def calcular_similitudbinarizado(news_corpus):
    archivo_prueba_path = archivo_prueba_entry.get()

    if not archivo_prueba_path:
        resultados_text.config(state=tk.NORMAL)
        resultados_text.delete(1.0, tk.END)
        resultados_text.insert(tk.END, "Por favor, seleccione un archivo de prueba primero.")
        resultados_text.config(state=tk.DISABLED)
        return

    with open(archivo_prueba_path, "r", encoding="utf-8") as archivo:
        new_document = archivo.read()
        new_document = limpiar_texto(new_document)  # Limpia el texto

    vectorizer_count = CountVectorizer(binary=True)
    X_count = vectorizer_count.fit_transform(news_corpus)

    new_document_binary = vectorizer_count.transform([new_document])

    similarities_binary = cosine_similarity(new_document_binary, X_count)

    most_similar_indices_binary = similarities_binary.argsort()[0][::-1][:10]

    # Muestra las noticias similares en el área de texto
    resultados_text.config(state=tk.NORMAL)
    resultados_text.delete(1.0, tk.END)
    for idx, similitud in zip(most_similar_indices_binary, similarities_binary[0][most_similar_indices_binary]):
        noticia_similar = news_corpus[idx]
        codigo = noticia_similar.split('&&&&&&&&')[0]
        linea_codigo = news_corpus.index(noticia_similar) + 1  # Número de línea de código
        resultados_text.insert(tk.END, f"Similitud Binarizado: {similitud:.2f} - Línea {linea_codigo}\n")
    resultados_text.config(state=tk.DISABLED)


# Función para calcular la similitud y mostrar resultados (Vectorizado)
def calcular_similitudvectorizado(news_corpus):
    archivo_prueba_path = archivo_prueba_entry.get()

    if not archivo_prueba_path:
        resultados_text.config(state=tk.NORMAL)
        resultados_text.delete(1.0, tk.END)
        resultados_text.insert(tk.END, "Por favor, seleccione un archivo de prueba primero.")
        resultados_text.config(state=tk.DISABLED)
        return

    with open(archivo_prueba_path, "r", encoding="utf-8") as archivo:
        new_document = archivo.read()
        new_document = limpiar_texto(new_document)  # Limpia el texto

    vectorizer_frequency = CountVectorizer()
    X_frequency = vectorizer_frequency.fit_transform(news_corpus)

    new_document_frequency = vectorizer_frequency.transform([new_document])

    similarities_frequency = cosine_similarity(new_document_frequency, X_frequency)

    most_similar_indices_frequency = similarities_frequency.argsort()[0][::-1][:10]

    # Muestra las noticias similares en el área de texto
    resultados_text.config(state=tk.NORMAL)
    resultados_text.delete(1.0, tk.END)
    for idx, similitud in zip(most_similar_indices_frequency,
                              similarities_frequency[0][most_similar_indices_frequency]):
        noticia_similar = news_corpus[idx]
        codigo = noticia_similar.split('&&&&&&&&')[0]
        linea_codigo = news_corpus.index(noticia_similar) + 1  # Número de línea de código
        resultados_text.insert(tk.END, f"Similitud Vectorizado: {similitud:.2f} - Línea {linea_codigo}\n")
    resultados_text.config(state=tk.DISABLED)


# Función para limpiar el texto de símbolos y caracteres especiales
def limpiar_texto(texto):
    texto_limpio = re.sub(r'[^\w\s]', '', texto)  # Elimina símbolos y caracteres especiales
    return texto_limpio


# Función para crear tablas y guardar en PDF
def crear_tablas(resultados):
    # Crear un documento PDF
    pdf_filename = "resultados_similitud.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Lista para almacenar las tablas
    elementos = []

    # Encabezado de la tabla
    encabezado = ["Documento de Prueba", "Tipo de Representación", "Número de Prueba", "Contenido de la Prueba",
                  "Valor de Similitud"]

    # Datos para la tabla
    datos = [encabezado]  # Agregar encabezado primero

    for resultado in resultados:
        documento_prueba, tipo_representacion, numero_prueba, contenido_prueba, valor_similitud = resultado
        fila = [documento_prueba, tipo_representacion, str(numero_prueba), contenido_prueba, f"{valor_similitud:.2f}"]
        datos.append(fila)

    # Crear la tabla
    tabla = Table(datos)

    # Estilo de la tabla
    estilo = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Fondo del encabezado
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Color de texto del encabezado
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Alineación del texto al centro
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Fuente en negrita
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Espaciado inferior del encabezado
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Fondo de las filas de datos
        ('GRID', (0, 0), (-1, -1), 1, colors.black)  # Líneas de la cuadrícula
    ])
    tabla.setStyle(estilo)

    # Agregar la tabla al documento
    elementos.append(tabla)

    # Construir el PDF
    doc.build(elementos)

    print(f"El archivo PDF '{pdf_filename}' ha sido creado con éxito.")


# Crear la ventana principal
ventana = tk.Tk()
ventana.config(width=1000, height=800)
ventana.title("Calculadora de Similitud de Documentos")
ventana.configure(bg='blue')  # Establece el color de fondo en rojo

# Botón para cargar el nuevo documento
cargar_documento_button = tk.Button(ventana, text="Cargar Documento", command=cargar_documento)
cargar_documento_button.pack(pady=10)

# Campo de entrada para el archivo de prueba
archivo_prueba_entry = tk.Entry(ventana, width=80)
archivo_prueba_entry.pack()

# Área de texto para el nuevo documento
new_document_entry = tk.Text(ventana, height=10, width=80)
new_document_entry.pack()

# Botón para calcular la similitud (TF-IDF)
calcular_similitud_button_tfidf = tk.Button(ventana, text="Calcular Similitud TF-IDF",
                                            command=lambda: calcular_similitudtfidf(news_corpus))
calcular_similitud_button_tfidf.pack(pady=10)

# Botón para calcular la similitud (Binarizado)
calcular_similitud_button_binarizado = tk.Button(ventana, text="Calcular Similitud Binarizado",
                                                 command=lambda: calcular_similitudbinarizado(news_corpus))
calcular_similitud_button_binarizado.pack(pady=10)

# Botón para calcular la similitud (Vectorizado)
calcular_similitud_button_vectorizado = tk.Button(ventana, text="Calcular Similitud Vectorizado",
                                                  command=lambda: calcular_similitudvectorizado(news_corpus))
calcular_similitud_button_vectorizado.pack(pady=10)

# Área de texto para mostrar los resultados
resultados_text = tk.Text(ventana, height=20, width=80)
resultados_text.pack()

# Cargar el corpus de noticias desde un archivo de texto
with open("corpus_noticias.txt", "r", encoding="utf-8") as file:
    news_corpus = file.read().splitlines()

# Botón para crear tablas y guardar en PDF
crear_tablas_button = tk.Button(ventana, text="Crear Tablas y Guardar en PDF", command=lambda: crear_tablas(resultados))
crear_tablas_button.pack(pady=10)

ventana.mainloop()

