#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Standard Python Libraries ---
# --- Standardowe Biblioteki Pythona ---

# For filesystem operations like creating paths.
# Do operacji na systemie plików, np. tworzenia ścieżek.
import os

# For loading configuration from YAML files.
# Do wczytywania konfiguracji z plików YAML.
import yaml

# For high-level file operations like deleting directory trees.
# Do operacji na plikach wysokiego poziomu, jak usuwanie drzew katalogów.
import shutil

# For creating and managing logs (to console and file).
# Do tworzenia i zarządzania logami (do konsoli i pliku).
import logging
from logging.handlers import RotatingFileHandler

# --- LangChain & Google Libraries ---
# --- Biblioteki LangChain i Google ---

# LangChain's specific integrations for Google's Generative AI models (LLM and Embeddings).
# Specyficzne integracje LangChain dla modeli Generative AI od Google (LLM i Embeddings).
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangChain's integration for the Chroma vector database.
# Integracja LangChain dla wektorowej bazy danych Chroma.
from langchain_chroma import Chroma

# A tool for creating reusable and structured prompts for the language model.
# Narzędzie do tworzenia reużywalnych i ustrukturyzowanych promptów dla modelu językowego.
from langchain.prompts import PromptTemplate

# A core component of LangChain Expression Language (LCEL) to pass inputs through the chain.
# Kluczowy komponent Języka Wyrażeń LangChain (LCEL) do przekazywania danych wejściowych w łańcuchu.
from langchain.schema.runnable import RunnablePassthrough

# A simple parser to convert the language model's output into a standard string.
# Prosty parser do konwersji wyniku modelu językowego na standardowy ciąg znaków.
from langchain.schema.output_parser import StrOutputParser

# --- Local Project Modules ---
# --- Lokalne Moduły Projektu ---

# Importing our custom function from the preprocessor script to get prepared documents.
# Importowanie naszej własnej funkcji ze skryptu preprocesora, aby pobrać przygotowane dokumenty.
from RAG.rag_data_preprocessor import get_documents_from_source


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
            "LLM_MODEL": rag_config['llm_model']
        }
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


def perform_legal_analysis(query: str, legal_retriever, case_retriever):
    """
    Performs a multi-step RAG query to synthesize facts and law.
    Wykonuje wieloetapowe zapytanie RAG w celu syntezy faktów i prawa.
    """
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

    final_template = """Jesteś ekspertem prawnym i analitykiem. Twoim zadaniem jest staranna analiza przedstawionego stanu faktycznego w świetle załączonych przepisów prawnych, aby odpowiedzieć na pytanie użytkownika. Odpowiedź musi być spójna, logiczna i odwoływać się zarówno do faktów, jak i do prawa, cytując źródła w nawiasach kwadratowych po każdej informacji, np. [źródło: KODEKS_KARNY.pdf, Art. 148].

PYTANIE UŻYTKOWNIKA:
{question}

ZEBRANE FAKTY (Z DOKUMENTÓW SPRAWY):
{factual_context}

ZEBRANE PRZEPISY PRAWNE:
{legal_context}

KOMPLEKSOWA ANALIZA PRAWNA:"""

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


if __name__ == "__main__":
    if not CONFIG:
        logger.critical("Configuration failed to load. Exiting.")
        print(
            "English: FATAL - Configuration failed to load. Exiting.\nPolski: BŁĄD KRYTYCZNY - Nie udało się wczytać konfiguracji. Zamykanie.")
        exit()

    # --- Rebuild Flags ---
    # Ustaw na True, jeśli chcesz wymusić przebudowanie konkretnej bazy
    # Set to True if you want to force a rebuild of a specific database
    REBUILD_LEGAL_DB = False
    REBUILD_CASE_DB = False

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