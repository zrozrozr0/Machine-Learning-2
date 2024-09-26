import prettytable
import tabulate
import re
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

with open('tweets.txt', 'r', encoding='utf-8') as archivo:
    tweets = archivo.read()

# Se definen los patrones de expresiones regulares
patron_hashtags = r'#\w+'
patron_usuarios = r'@\w+'
patron_tiempo = r'\b(?:\d{1,2}(?::\d{2})?[apmAPM]{2}|(?:[0-1]?[0-9]|2[0-3]):[0-5][0-9])\b'
patron_emoticonos = r'[:;=][-^]?[DPp3)(\]\[oO/\|3cC*}{>_<]|<3|xd|Xd|xD|XD|\^-?\^|;-?\)|:-?D|:-?P|:-?\(|:-?\/|<3\b'
patron_emojis_unicode = r'[\U0001f600-\U0001f9cf]'
# Se encuentran coincidencias
hashtags = re.findall(patron_hashtags, tweets)
usuarios = re.findall(patron_usuarios, tweets)
coincidencias_tiempo = re.findall(patron_tiempo, tweets)
emoticonos = re.findall(patron_emoticonos, tweets)
emojis_unicode = re.findall(patron_emojis_unicode, tweets)
# Se implementa Counter para contar la frecuencia de cada tipo de cadena
n_hashtags = Counter(hashtags)
n_usuarios = Counter(usuarios)
n_tiempo = Counter(coincidencias_tiempo)
n_emoticonos = Counter(emoticonos)
n_emojis_unicode = Counter(emojis_unicode)


top10_hashtags = n_hashtags.most_common(10)
top10_usuarios = n_usuarios.most_common(10)
top10_tiempo = n_tiempo.most_common(10)
top10_emoticonos = n_emoticonos.most_common(10)
top10_emojis_unicode = n_emojis_unicode.most_common(10)


data = [("Elemento (Hashtags)", "Frecuencia de elemento ", "Top 10 elemento"),
        *[("Hashtags", freq, hashtag) for hashtag, freq in top10_hashtags],
        ("Elemento (Usuarios)", "Frecuencia de elemento", "Top 10 elemento"),
        *[("Usuarios", freq, usuario) for usuario, freq in top10_usuarios],
        ("Elemento (Tiempo)", "Frecuencia de elemento", "Top 10 elemento"),
        *[("Tiempo", freq, tiempo) for tiempo, freq in top10_tiempo],
        ("Elemento (Emoticonos en ASCII)", "Frecuencia de Elemento", "Top 10 elemento"),
        *[("Emoticonos en ASCII", freq, emoticono) for emoticono, freq in top10_emoticonos],
        ("Elemento (Emoji Unicode)", "Frecuencia de Elemento", "Top 10 elemento"),
        *[("Emoji Unicode", freq, emojis_unicode) for emojis_unicode, freq in top10_emojis_unicode]
       ]

# Definir estilos para el documento PDF
styles = getSampleStyleSheet()
style_table = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ('FONTNAME', (0, 1), (-1, -1), 'Segoe UI Emoji'),  # Cambiar la fuente aquí
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
])

# Crear el documento PDF y la tabla
output_file = "Tabla_Datos_Tweets.pdf"
doc = SimpleDocTemplate(output_file, pagesize=letter)
table = Table(data)

# Aplicar estilos a la tabla
table.setStyle(style_table)

# Construir el contenido y guardar el PDF
content = [table]
doc.build(content)

print(f"El archivo PDF '{output_file}' ha sido creado con éxito.")
