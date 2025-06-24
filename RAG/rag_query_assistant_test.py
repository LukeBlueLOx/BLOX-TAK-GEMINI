#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Standard Python Libraries ---
# --- Standardowe Biblioteki Pythona ---
import os
import yaml
import shutil
import logging
from logging.handlers import RotatingFileHandler
import datetime
import re

# --- PDF Generation Libraries ---
# --- Biblioteki do generowania PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_JUSTIFY

# --- LangChain & Google Libraries ---
# --- Biblioteki LangChain i Google ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Local Project Modules ---
# --- Lokalne Moduły Projektu ---
from rag_data_preprocessor import get_documents_from_source


def setup_logger(log_folder: str):
    """
    Sets up a logger to output to both console and a rotating file.
    Konfiguruje logger do zapisu zarówno do konsoli, jak i do rotacyjnego pliku.
    """
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
    """
    Loads the main configuration file.
    Wczytuje główny plik konfiguracyjny.
    """
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
            "EMBEDDING_MODEL": rag_config['embedding_model'],
            "LLM_MODEL": rag_config['llm_model'],
            "OUTPUT_FOLDER": os.path.join(base_path, "PROCESSED_OUTPUT"),
            "FONT_PATH": os.path.join(base_path, "UbuntuMono-Regular.ttf"),
            "FONT_NAME": "UbuntuMono",
            "FONT_SIZE": 10
        }
        os.makedirs(conf["OUTPUT_FOLDER"], exist_ok=True)
        print("English: Configuration loaded successfully.\nPolski: Konfiguracja załadowana pomyślnie.")
        return conf
    except Exception as e:
        print(
            f"English: FATAL ERROR loading configuration from {config_path}: {e}\nPolski: KRYTYCZNY BŁĄD podczas ładowania konfiguracji z {config_path}: {e}")
        return None


# --- Global Initialization ---
# --- Globalna Inicjalizacja ---
CONFIG = load_configuration()
if CONFIG:
    logger = setup_logger(CONFIG['LOG_FOLDER'])
    LLM = GoogleGenerativeAI(model=CONFIG['LLM_MODEL'], google_api_key=CONFIG['GOOGLE_API_KEY'])
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=CONFIG['EMBEDDING_MODEL'], google_api_key=CONFIG['GOOGLE_API_KEY'])

    try:
        pdfmetrics.registerFont(TTFont(CONFIG['FONT_NAME'], CONFIG['FONT_PATH']))
        print(f"Font '{CONFIG['FONT_NAME']}' registered successfully.")
    except Exception as e:
        print(f"ERROR: Could not register font. Defaulting to Helvetica. Error: {e}")
        CONFIG['FONT_NAME'] = "Helvetica"

else:
    logger = setup_logger('LOGS')


def initialize_vector_store(db_path: str, source_folder: str, corpus_type: str,
                            force_rebuild: bool = False) -> Chroma | None:
    """
    Initializes a Chroma vector store. Builds it if it doesn't exist or if rebuild is forced.
    Inicjalizuje bazę wektorową Chroma. Buduje ją, jeśli nie istnieje lub wymuszono przebudowę.
    """
    if os.path.exists(db_path) and not force_rebuild:
        logger.info(f"Loading existing vector database from: {db_path}")
        print(
            f"English: Loading existing vector database from: {db_path}\nPolski: Ładuję istniejącą bazę wektorową z: {db_path}")
        try:
            db = Chroma(persist_directory=db_path, embedding_function=EMBEDDINGS)
            return db
        except Exception as e:
            logger.error(
                f"Failed to load existing database at {db_path}. It might be corrupted. Try rebuilding. Error: {e}")
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
    """
    Helper function to format retrieved documents for the prompt, including metadata.
    Funkcja pomocnicza do formatowania pobranych dokumentów na potrzeby promptu, włączając metadane.
    """
    formatted_docs = []
    for doc in docs:
        if doc.metadata.get("type") == "legal_corpus":
            header = f"Fragment z prawa: {doc.metadata.get('source', 'b.d.')}, Artykuł: {doc.metadata.get('article', 'b.d.')}"
        elif doc.metadata.get("type") == "case_corpus":
            header = f"Fragment z dowodu: {doc.metadata.get('source_original', 'b.d.')} (w archiwum: {doc.metadata.get('source_archive', 'b.d.')})"
        else:
            header = "Fragment z nieznanego źródła"

        content = doc.page_content
        formatted_docs.append(f"{header}\n---\n{content}\n---\n")
    return "\n".join(formatted_docs)


def reflow_text(text: str) -> str:
    """
    Intelligently joins lines for better PDF formatting.
    Inteligentnie łączy linie dla lepszego formatowania PDF.
    """
    reflowed = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    reflowed = re.sub(r'\n{2,}', '<br/><br/>', reflowed)
    return reflowed


def save_text_to_pdf(text_content, output_pdf_path, font_name, font_size):
    """
    Saves the given text content to a PDF file using Platypus for proper text wrapping.
    Zapisuje podany tekst do pliku PDF, używając biblioteki Platypus do poprawnego zawijania tekstu.
    """
    print(f"English: Saving PDF to: {output_pdf_path}...\nPolski: Zapisuję PDF do: {output_pdf_path}...")

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    style = ParagraphStyle(name='Normal_Justified', parent=styles['Normal'], fontName=font_name, fontSize=font_size,
                           leading=font_size * 1.5, alignment=TA_JUSTIFY)

    processed_text = reflow_text(text_content)
    story = [Paragraph(processed_text, style)]

    try:
        doc.build(story)
        print(f"English: Successfully saved PDF: {output_pdf_path}\nPolski: Pomyślnie zapisano PDF: {output_pdf_path}")
    except Exception as e:
        print(f"English: ERROR saving PDF: {e}\nPolski: BŁĄD podczas zapisywania PDF: {e}")


def perform_legal_analysis(query: str, legal_retriever, case_retriever):
    """
    Performs a multi-step RAG query to synthesize facts and law.
    Wykonuje wieloetapowe zapytanie RAG w celu syntezy faktów i prawa.
    """
    now = datetime.datetime.now()
    full_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    logger.info(f"Performing full legal analysis for query: '{query[:80]}...'")
    print(
        f"English: Performing full legal analysis for query: '{query[:80]}...'\nPolski: Wykonuję pełną analizę prawną dla zapytania: '{query[:80]}...'")

    logger.info("Step 1: Retrieving facts from case_db...")
    print(
        "English: Step 1: Retrieving facts from case database...\nPolski: Krok 1: Pobieram fakty z bazy danych sprawy...")
    factual_context_docs = case_retriever.invoke(query)
    factual_context_str = format_docs(factual_context_docs)

    logger.info("Step 2: Retrieving law from legal_db...")
    print(
        "English: Step 2: Retrieving law from legal database...\nPolski: Krok 2: Pobieram prawo z bazy danych prawnej...")
    legal_context_docs = legal_retriever.invoke(query)
    legal_context_str = format_docs(legal_context_docs)

    logger.info("Step 3: Synthesizing final answer with LLM...")
    print(
        "English: Step 3: Synthesizing final answer with LLM...\nPolski: Krok 3: Syntezuję ostateczną odpowiedź za pomocą LLM...")

    # Zmodyfikowany szablon, który instruuje model, aby dołączył tłumaczenie i metadane
    final_template = f"""Jesteś ekspertem prawnym i analitykiem. Twoim zadaniem jest staranna analiza przedstawionego stanu faktycznego w świetle załączonych przepisów prawnych, aby odpowiedzieć na pytanie użytkownika.
Odpowiedź musi być spójna, logiczna i odwoływać się zarówno do faktów, jak i do prawa, cytując źródła w nawiasach kwadratowych po każdej informacji, np. [źródło: KODEKS_KARNY.pdf, Art. 148].

Struktura odpowiedzi musi być następująca:
1.  **Analiza Prawna (Język Polski):** Na początku przedstaw kompletną analizę w języku polskim.
2.  **English Translation:** Następnie, pod nagłówkiem "--- ENGLISH TRANSLATION ---", przetłumacz całą powyższą analizę na język angielski.
3.  **Metadata:** Na samym końcu odpowiedzi, pod nagłówkiem "--- METADATA ---", dodaj stopkę z metadanymi w następującym formacie:
    Użyty prompt (PL): [Wklej tutaj treść całego tego promptu, zaczynając od "Jesteś ekspertem prawnym..."]
    Used prompt (EN): [Wklej tutaj dokładne tłumaczenie całego tego promptu na język angielski]
    Model: {CONFIG['LLM_MODEL']}
    Timestamp: {full_timestamp}

---
PYTANIE UŻYTKOWNIKA:
{{question}}

ZEBRANE FAKTY (Z DOKUMENTÓW SPRAWY):
{{factual_context}}

ZEBRANE PRZEPISY PRAWNE:
{{legal_context}}

---
KOMPLEKSOWA ODPOWIEDŹ (wygeneruj zgodnie z wymaganą strukturą):
"""

    prompt = PromptTemplate.from_template(final_template)

    analysis_chain = prompt | LLM | StrOutputParser()

    response = analysis_chain.invoke({
        "question": query,
        "factual_context": factual_context_str,
        "legal_context": legal_context_str
    })

    print("\n" + "=" * 25 + " WYNIK ANALIZY " + "=" * 25)
    print(response)
    print("=" * 65 + "\n")

    # Zapisz wynik do pliku PDF
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"RAG_Analysis_{timestamp}.pdf"
    output_path = os.path.join(CONFIG['OUTPUT_FOLDER'], output_filename)

    save_text_to_pdf(
        text_content=response,
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

    # --- Rebuild Flags ---
    # Ustaw na True, jeśli chcesz wymusić przebudowanie konkretnej bazy
    # Set to True if you want to force a rebuild of a specific database
    REBUILD_LEGAL_DB = True
    REBUILD_CASE_DB = True

    legal_db = initialize_vector_store(
        db_path=CONFIG['LEGAL_DB'],
        source_folder=CONFIG['LEGAL_SOURCE'],
        corpus_type='legal',
        force_rebuild=REBUILD_LEGAL_DB
    )

    case_db = initialize_vector_store(
        db_path=CONFIG['CASE_DB'],
        source_folder=CONFIG['CASE_SOURCE'],
        corpus_type='case',
        force_rebuild=REBUILD_CASE_DB
    )

    if not legal_db or not case_db:
        logger.critical("One or more databases failed to initialize. Cannot proceed with queries.")
        print(
            "English: CRITICAL - One or more databases failed to initialize. Cannot proceed.\nPolski: BŁĄD KRYTYCZNY - Inicjalizacja jednej lub więcej baz danych nie powiodła się. Nie można kontynuować.")
    else:
        legal_retriever = legal_db.as_retriever(search_kwargs={'k': 5})
        case_retriever = case_db.as_retriever(search_kwargs={'k': 8})

        # --- Ask your question here ---
        # --- Zadaj swoje pytanie tutaj ---
        user_question = "Jakie prawa gwarantowane przez Konstytucję RP mogły zostać naruszone, biorąc pod uwagę treść moich pism, w których opisuję brak środków do życia i problemy zdrowotne?"

        perform_legal_analysis(
            query=user_question,
            legal_retriever=legal_retriever,
            case_retriever=case_retriever
        )