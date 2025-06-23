# limpieza.py
import re
import unicodedata
import pandas as pd
import spacy
from nltk.corpus import stopwords
import stopwordsiso
from fuzzywuzzy import fuzz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker

# --- Inicialización ---
tqdm.pandas()
nlp = spacy.load('es_core_news_sm')
stemmer = SnowballStemmer('spanish')
spell = SpellChecker(language='es')

# --- Stopwords ---
spacy_stopwords = nlp.Defaults.stop_words
nltk_stopwords = stopwords.words('spanish')
iso_stopwords = stopwordsiso.stopwords('es')
combined_stopwords = set(spacy_stopwords).union(nltk_stopwords, iso_stopwords)

# combined_stopwords = set(spacy_stopwords)

# --- Conectores y muletillas ---
conectores_y_muletillas = {
    # Conectores discursivos comunes
    "ademas", "tambien", "por eso", "por lo tanto", "asi que",
    "sin embargo", "no obstante", "entonces", "de hecho",
    "por ejemplo", "en cambio", "aunque", "es decir",
    "en resumen", "por consiguiente", "por ende", "por otra parte",
    "por un lado", "por otro lado", "por supuesto", "en conclusion",
    "al contrario", "en realidad", "en efecto", "por tanto",

    # Muletillas comunes del habla oral
    "eh", "este", "pues", "o sea", "verdad", "bueno",
    "miren", "imaginense", "fijense", "digamos", "como tal",
    "entonces", "claro", "ok", "a ver", "vean",
    "nada mas", "eso si", "ahora bien", "asi como", "mas bien",
    "si claro", "me explico", "vale", "verdad", "ya saben",

    # Repeticiones y expresiones vacías
    "y bueno", "entonces bueno", "entonces este", "y este",
    "bueno pues", "o sea que", "lo que pasa", "como que",
    "este pues", "pues bueno"
}

# --- Limpieza básica ---
def quitar_acentos(texto):
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in texto if not unicodedata.combining(c))

def solo_letras_espacios(texto):
    return re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]', '', texto)

def eliminar_espacios_multiples(texto):
    return re.sub(r'\s+', ' ', texto.strip())

def eliminar_palabras_similares(texto, umbral=70):
    palabras = texto.split()
    unicas = []
    for palabra in palabras:
        if not any(fuzz.ratio(palabra, p) > umbral for p in unicas):
            unicas.append(palabra)
    return ' '.join(unicas)

# --- Función principal ---
def limpiar_texto(texto):
    texto = texto.lower()
    texto = quitar_acentos(texto)
    texto = solo_letras_espacios(texto)
    texto = eliminar_espacios_multiples(texto)

    doc = nlp(texto)
    tokens = [
        token.text for token in doc
        if token.text not in combined_stopwords
        and token.text not in conectores_y_muletillas
        and len(token.text) >= 4
    ]

    texto_limpio = ' '.join(tokens)

    # Opcional: eliminar palabras similares
    return eliminar_palabras_similares(texto_limpio, umbral=90)
    # return texto_limpio

# --- Procesamiento en paralelo ---
def procesar_en_paralelo_tqdm(serie_texto):
    with Pool(cpu_count()) as pool:
        resultados = list(tqdm(pool.imap_unordered(limpiar_texto, serie_texto), total=len(serie_texto)))
    return resultados

# --- Punto de entrada ---
if __name__ == "__main__":
    df = pd.read_excel("./scripts/conferencias_original.xlsx")
    df["texto_limpio"] = procesar_en_paralelo_tqdm(df["texto"])
    df.to_excel("./scripts/conferencias_limpias.xlsx", index=False)
