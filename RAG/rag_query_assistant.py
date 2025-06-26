#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================================================
# === IMPORTOWANE BIBLIOTEKI Z OPISAMI / IMPORTED LIBRARIES WITH DESCRIPTIONS ===
# =====================================================================================

# --- Standardowe Biblioteki Pythona (Standard Python Libraries) ---

# Provides functions for interacting with the operating system.
# Zapewnia funkcje do interakcji z systemem operacyjnym.
import os

# Allows parsing configuration files in YAML format.
# Umożliwia parsowanie plików konfiguracyjnych w formacie YAML.
import yaml

# Offers high-level file and directory operations.
# Oferuje wysokopoziomowe operacje na plikach i folderach.
import shutil

# The primary module for logging information.
# Podstawowy moduł do logowania informacji.
import logging

# A handler for logging to a rotating set of files.
# Handler do logowania do rotacyjnego zestawu plików.
from logging.handlers import RotatingFileHandler

# Allows working with dates and times.
# Pozwala na pracę z datami i czasem.
import datetime

# A module for working with Regular Expressions.
# Moduł do pracy z wyrażeniami regularnymi.
import re

# --- Biblioteki do generowania PDF (PDF Generation Libraries) ---

# Provides a constant defining the standard A4 page size.
# Dostarcza stałą definiującą standardowy rozmiar strony A4.
from reportlab.lib.pagesizes import A4

# Core classes for building the PDF document.
# Kluczowe klasy do budowania dokumentu PDF.
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Tools for text styling.
# Narzędzia do stylizacji tekstu.
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Allows loading custom fonts from .ttf files.
# Umożliwia załadowanie niestandardowych czcionek z plików .ttf.
from reportlab.pdfbase.ttfonts import TTFont

# Allows registering loaded fonts.
# Pozwala na rejestrowanie załadowanych czcionek.
from reportlab.pdfbase import pdfmetrics

# Provides a constant for justified text alignment.
# Dostarcza stałą dla justowania tekstu.
from reportlab.lib.enums import TA_JUSTIFY

# --- Biblioteki LangChain i Google (LangChain & Google Libraries) ---

# LangChain integrations with Google's generative and embedding models.
# Integracje LangChain z modelami generatywnymi i embeddingowymi Google.
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangChain integration with the Chroma vector database.
# Integracja LangChain z wektorową bazą danych Chroma.
from langchain_chroma import Chroma

# A class for creating dynamic prompt templates.
# Klasa do tworzenia dynamicznych szablonów promptów.
from langchain.prompts import PromptTemplate

# A component for passing inputs through a chain.
# Komponent do przekazywania danych wejściowych przez łańcuch.
from langchain.schema.runnable import RunnablePassthrough

# A parser to convert model output to a string.
# Parser konwertujący odpowiedź modelu na ciąg znaków.
from langchain.schema.output_parser import StrOutputParser

# --- Lokalne Moduły Projektu (Local Project Modules) ---

# Assuming rag_data_preprocessor.py exists with the required function.
# Zakładając, że plik rag_data_preprocessor.py istnieje z wymaganą funkcją.
from rag_data_preprocessor import get_documents_from_source


# =====================================================================================
# === GŁÓWNA CZĘŚĆ SKRYPTU / MAIN SCRIPT BODY ===
# =====================================================================================

def setup_logger(log_folder: str):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = RotatingFileHandler(os.path.join(log_folder, 'rag_assistant.log'), maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_configuration(config_path="scripts/config.yaml"):
    # Symulacja dla działania samodzielnego skryptu
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    if not os.path.exists(config_path):
        print(
            f"English: Config file not found. Creating example in '{config_path}'.\nPolski: Plik konfiguracyjny nie istnieje. Tworzę przykładowy w '{config_path}'.")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({
                'KEY': 'TWOJ_KLUCZ_GOOGLE_API',  # <--- WSTAW TUTAJ SWÓJ KLUCZ!
                'base_path': '.',
                'rag_pipeline_config': {
                    'legal_source_folder': 'LEGAL_SOURCE', 'case_source_folder': 'CASE_SOURCE',
                    'vector_db_legal_folder': 'VECTOR_DB_LEGAL', 'vector_db_case_folder': 'VECTOR_DB_CASE',
                    'log_folder': 'LOGS', 'embedding_model': 'models/embedding-001',
                    'llm_model': 'gemini-1.5-pro-latest'
                }}, f)
        os.makedirs("LEGAL_SOURCE", exist_ok=True)
        os.makedirs("CASE_SOURCE", exist_ok=True)
    try:
        with open(config_path, "r", encoding="utf-8") as cr:
            config = yaml.full_load(cr)
        base_path = config['base_path']
        rag_config = config['rag_pipeline_config']
        conf = {
            "GOOGLE_API_KEY": config['KEY'],
            "LEGAL_SOURCE": os.path.join(base_path, rag_config['legal_source_folder']),
            "CASE_SOURCE": os.path.join(base_path, rag_config['case_source_folder']),
            "LEGAL_DB": os.path.join(base_path, rag_config['vector_db_legal_folder']),
            "CASE_DB": os.path.join(base_path, rag_config['vector_db_case_folder']),
            "LOG_FOLDER": os.path.join(base_path, rag_config['log_folder']),
            "EMBEDDING_MODEL": rag_config['embedding_model'], "LLM_MODEL": rag_config['llm_model'],
            "OUTPUT_FOLDER": os.path.join(base_path, "PROCESSED_OUTPUT"),
            "FONT_PATH": os.path.join(base_path, "UbuntuMono-Regular.ttf"), "FONT_NAME": "UbuntuMono", "FONT_SIZE": 11
        }
        os.makedirs(conf["OUTPUT_FOLDER"], exist_ok=True)
        print("English: Configuration loaded successfully.\nPolski: Konfiguracja załadowana pomyślnie.")
        return conf
    except Exception as e:
        print(
            f"English: FATAL ERROR loading configuration from {config_path}: {e}\nPolski: KRYTYCZNY BŁĄD podczas ładowania konfiguracji z {config_path}: {e}")
        return None


CONFIG = load_configuration()
if CONFIG:
    if 'TWOJ_KLUCZ_GOOGLE_API' in CONFIG.get('GOOGLE_API_KEY', ''):
        print(
            "\nEnglish: WARNING: Using default API key. Please replace 'TWOJ_KLUCZ_GOOGLE_API' in 'scripts/config.yaml'.\nPolski: OSTRZEŻENIE: Używasz domyślnego klucza API. Zastąp 'TWOJ_KLUCZ_GOOGLE_API' w 'scripts/config.yaml'.\n")
    logger = setup_logger(CONFIG['LOG_FOLDER'])
    LLM = GoogleGenerativeAI(model=CONFIG['LLM_MODEL'], google_api_key=CONFIG['GOOGLE_API_KEY'])
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=CONFIG['EMBEDDING_MODEL'], google_api_key=CONFIG['GOOGLE_API_KEY'])
    try:
        if os.path.exists(CONFIG['FONT_PATH']):
            pdfmetrics.registerFont(TTFont(CONFIG['FONT_NAME'], CONFIG['FONT_PATH']))
            print(
                f"English: Font '{CONFIG['FONT_NAME']}' registered successfully.\nPolski: Czcionka '{CONFIG['FONT_NAME']}' zarejestrowana pomyślnie.")
        else:
            print(
                f"English: WARNING: Font file not found at {CONFIG['FONT_PATH']}. Defaulting to Helvetica.\nPolski: OSTRZEŻENIE: Plik czcionki nie znaleziony w {CONFIG['FONT_PATH']}. Używam domyślnej Helvetica.")
            CONFIG['FONT_NAME'] = "Helvetica"
    except Exception as e:
        print(
            f"English: ERROR registering font. Defaulting to Helvetica. Error: {e}\nPolski: BŁĄD podczas rejestracji czcionki. Używam domyślnej Helvetica. Błąd: {e}")
        CONFIG['FONT_NAME'] = "Helvetica"
else:
    logger = setup_logger('LOGS')


def initialize_vector_store(db_path: str, source_folder: str, corpus_type: str,
                            force_rebuild: bool = False) -> Chroma | None:
    if os.path.exists(db_path) and not force_rebuild:
        logger.info(f"Loading existing vector database from: {db_path}")
        print(
            f"English: Loading existing vector database from: {db_path}\nPolski: Ładuję istniejącą bazę wektorową z: {db_path}")
        try:
            return Chroma(persist_directory=db_path, embedding_function=EMBEDDINGS)
        except Exception as e:
            logger.error(f"Failed to load database at {db_path}. It might be corrupted. Try rebuilding. Error: {e}")
            print(
                f"English: ERROR - Failed to load existing database at {db_path}. Try rebuilding. Error: {e}\nPolski: BŁĄD - Nie udało się załadować istniejącej bazy z {db_path}. Spróbuj ją przebudować. Błąd: {e}")
            return None
    logger.info(f"Rebuilding vector database for corpus '{corpus_type}'...")
    print(
        f"English: Rebuilding vector database for corpus '{corpus_type}'...\nPolski: Przebudowuję bazę wektorową dla korpusu '{corpus_type}'...")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    documents = get_documents_from_source(source_folder, corpus_type)
    if not documents:
        logger.error(f"No documents found for corpus '{corpus_type}'. Database not built.")
        print(
            f"English: ERROR - No documents found for corpus '{corpus_type}'. Database not built.\nPolski: BŁĄD - Nie znaleziono dokumentów dla korpusu '{corpus_type}'. Baza danych nie została zbudowana.")
        return None
    db = Chroma.from_documents(documents, EMBEDDINGS, persist_directory=db_path)
    logger.info(f"Successfully built and saved vector database at: {db_path}")
    print(
        f"English: Successfully built and saved vector database at: {db_path}\nPolski: Pomyślnie zbudowano i zapisano bazę wektorową w: {db_path}")
    return db


def format_docs(docs: list) -> str:
    formatted_docs = []
    for doc in docs:
        if doc.metadata.get("type") == "legal_corpus":
            header = f"Fragment z prawa: {doc.metadata.get('source', 'b.d.')}"
        elif doc.metadata.get("type") == "case_corpus":
            header = f"Fragment z dowodu: {doc.metadata.get('source_original', 'b.d.')}"
        else:
            header = f"Fragment z: {doc.metadata.get('source', 'nieznane źródło')}"
        content = doc.page_content
        formatted_docs.append(f"{header}\n---\n{content}\n---\n")
    return "\n".join(formatted_docs)


def reflow_text(text: str) -> str:
    reflowed = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    reflowed = re.sub(r'\n{2,}', '<br/><br/>', reflowed)
    return reflowed


def save_text_to_pdf(text_content, output_pdf_path, font_name, font_size):
    print(f"English: Saving PDF to: {output_pdf_path}...\nPolski: Zapisuję PDF do: {output_pdf_path}...")
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    style = ParagraphStyle(name='Normal_Justified', parent=styles['Normal'], fontName=font_name, fontSize=font_size,
                           leading=font_size * 1.5, alignment=TA_JUSTIFY)
    story = [Paragraph(text_content, style)]
    try:
        doc.build(story)
        print(f"English: Successfully saved PDF: {output_pdf_path}\nPolski: Pomyślnie zapisano PDF: {output_pdf_path}")
    except Exception as e:
        print(f"English: ERROR saving PDF: {e}\nPolski: BŁĄD podczas zapisywania PDF: {e}")


# ZMIANA: Funkcja przyjmuje teraz dwie wersje językowe zapytania.
# CHANGE: The function now accepts two language versions of the query.
def perform_legal_analysis(query_pl: str, query_en: str, legal_retriever, case_retriever):
    now = datetime.datetime.now()
    full_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    logger.info(f"Performing full legal analysis for query: '{query_pl[:80]}...'")
    print(
        f"English: Performing full legal analysis for query: '{query_pl[:80]}...'\nPolski: Wykonuję pełną analizę prawną dla zapytania: '{query_pl[:80]}...'")

    # Analiza i pobieranie danych bazuje na polskim zapytaniu
    logger.info("Step 1: Retrieving facts...")
    print(
        "English: Step 1: Retrieving facts from case database...\nPolski: Krok 1: Pobieram fakty z bazy danych sprawy...")
    factual_context_docs = case_retriever.invoke(query_pl)
    factual_context_str = format_docs(factual_context_docs)

    logger.info("Step 2: Retrieving law...")
    print(
        "English: Step 2: Retrieving law from legal database...\nPolski: Krok 2: Pobieram prawo z bazy danych prawnej...")
    legal_context_docs = legal_retriever.invoke(query_pl)
    legal_context_str = format_docs(legal_context_docs)

    logger.info("Step 3: Synthesizing final answer with LLM...")
    print(
        "English: Step 3: Synthesizing final answer with LLM...\nPolski: Krok 3: Syntezuję ostateczną odpowiedź za pomocą LLM...")

    # Krok A: Uproszczony prompt po polsku
    final_template_pl = """Jesteś ekspertem prawnym i analitykiem. Twoim zadaniem jest staranna analiza przedstawionego stanu faktycznego w świetle załączonych przepisów prawnych, aby odpowiedzieć na pytanie użytkownika.
Odpowiedź musi być spójna, logiczna i odwoływać się zarówno do faktów, jak i do prawa, cytując źródła w nawiasach kwadratowych po każdej informacji, np. [źródło: KODEKS_KARNY.pdf, Art. 148].

STRUKTURA ODPOWIEDZI:
1.  **Analiza Prawna (Język Polski):** Na początku przedstaw kompletną analizę w języku polskim.
2.  **English Translation:** Następnie, pod nagłówkiem "--- ENGLISH TRANSLATION ---", przetłumacz całą powyższą analizę na język angielski.

---
PYTANIE UŻYTKOWNIKA:
{question}

ZEBRANE FAKTY (Z DOKUMENTÓW SPRAWY):
{factual_context}

ZEBRANE PRZEPISY PRAWNE:
{legal_context}

---
KOMPLEKSOWA ODPOWIEDŹ (wygeneruj zgodnie z wymaganą strukturą, zawierającą tylko Analizę i Tłumaczenie):
"""
    prompt = PromptTemplate.from_template(final_template_pl)
    analysis_chain = prompt | LLM | StrOutputParser()

    # Krok B: Otrzymujemy CZYSTĄ odpowiedź od modelu. Używamy polskiego zapytania.
    llm_response = analysis_chain.invoke({
        "question": query_pl,
        "factual_context": factual_context_str,
        "legal_context": legal_context_str
    })

    print("\n" + "=" * 25 + " ODPOWIEDŹ Z MODELU " + "=" * 25)
    print(llm_response)

    # Krok C: Przetwarzamy odpowiedź modelu do formatu PDF.
    processed_llm_response = reflow_text(llm_response)

    # Krok D: Skrypt Python BUDUJE OBA bloki metadanych.

    # Przetłumaczony szablon promptu na potrzeby angielskich metadanych
    final_template_en = """You are a legal expert and analyst. Your task is to carefully analyze the presented factual state in light of the attached legal provisions to answer the user's question.
The response must be coherent, logical, and refer to both the facts and the law, citing sources in square brackets after each piece of information, e.g., .

RESPONSE STRUCTURE:
1.  **Legal Analysis (Polish Language):** First, present a complete analysis in Polish.
2.  **English Translation:** Next, under the heading "--- ENGLISH TRANSLATION ---", translate the entire analysis above into English.

---
USER'S QUESTION:
{question}

COLLECTED FACTS (FROM CASE DOCUMENTS):
{factual_context}

COLLECTED LEGAL PROVISIONS:
{legal_context}

---
COMPREHENSIVE RESPONSE (generate according to the required structure, containing only the Analysis and Translation):
"""

    # Polski blok metadanych
    metadata_pl = f"""
    \n
--- METADATA (PL) ---
Zapytanie Użytkownika: {query_pl}
Model: {CONFIG['LLM_MODEL']}
Timestamp: {full_timestamp}
Użyty prompt:
---
{final_template_pl}
---
"""
    # Angielski blok metadanych - używa query_en
    metadata_en = f"""
--- METADATA (EN) ---
User Query: {query_en}
Model: {CONFIG['LLM_MODEL']}
Timestamp: {full_timestamp}
Used Prompt:
---
{final_template_en}
---
"""

    # Formatujemy oba bloki dla PDF (zamiana \n na <br/>).
    metadata_pl_for_pdf = metadata_pl.replace('\n', '<br/>')
    metadata_en_for_pdf = metadata_en.replace('\n', '<br/>')

    print("\n" + "=" * 25 + " WYGENEROWANE METADANE (KONSOLA) " + "=" * 25)
    print(metadata_pl)

    # Krok E: Łączymy wszystko w finalną treść dla PDF.
    final_pdf_content = processed_llm_response + metadata_pl_for_pdf + metadata_en_for_pdf

    # Krok F: Zapisujemy kompletną treść do PDF.
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_filename = f"RAG_Analysis_{timestamp}.pdf"
    output_path = os.path.join(CONFIG['OUTPUT_FOLDER'], output_filename)

    save_text_to_pdf(
        text_content=final_pdf_content,
        output_pdf_path=output_path,
        font_name=CONFIG['FONT_NAME'],
        font_size=CONFIG['FONT_SIZE']
    )


if __name__ == "__main__":
    if not CONFIG:
        logger.critical("Configuration failed to load. Exiting.")
        print(
            "English: FATAL - Configuration failed to load. Exiting.\nPolski: BŁĄD KRYTYCZNY - Nie udało się wczytać konfiguracji. Zamykanie.")
        exit()

    REBUILD_LEGAL_DB = False
    REBUILD_CASE_DB = False

    legal_db = initialize_vector_store(CONFIG['LEGAL_DB'], CONFIG['LEGAL_SOURCE'], 'legal', REBUILD_LEGAL_DB)
    case_db = initialize_vector_store(CONFIG['CASE_DB'], CONFIG['CASE_SOURCE'], 'case', REBUILD_CASE_DB)

    if not legal_db or not case_db:
        logger.critical("One or more databases failed to initialize. Cannot proceed with queries.")
        print(
            "English: CRITICAL - One or more databases failed to initialize. Cannot proceed.\nPolski: BŁĄD KRYTYCZNY - Inicjalizacja jednej lub więcej baz danych nie powiodła się. Nie można kontynuować.")
    else:
        legal_retriever = legal_db.as_retriever(search_kwargs={'k': 5})
        case_retriever = case_db.as_retriever(search_kwargs={'k': 8})

        # --- Zdefiniuj swoje pytania tutaj ---
        # --- Define your questions here ---
        user_question_pl = "Jakie prawa gwarantowane przez Konstytucję RP mogły zostać naruszone, biorąc pod uwagę treść moich pism, w których opisuję brak środków do życia i problemy zdrowotne?"
        user_question_en = "What rights guaranteed by the Constitution of the Republic of Poland could have been violated, considering the content of my letters in which I describe a lack of means of subsistence and health problems?"

        if 'TWOJ_KLUCZ_GOOGLE_API' in CONFIG.get('GOOGLE_API_KEY', ''):
            print(
                "English: CRITICAL ERROR: Cannot run analysis. Please enter your Google API key in 'scripts/config.yaml'.\nPolski: BŁĄD KRYTYCZNY: Nie można uruchomić analizy. Wprowadź swój klucz Google API w pliku 'scripts/config.yaml'.")
        else:
            # Przekazujemy obie wersje językowe pytania do funkcji
            perform_legal_analysis(user_question_pl, user_question_en, legal_retriever, case_retriever)