# --- Importing necessary libraries ---
# --- Importowanie niezbędnych bibliotek ---
import os  # Do operacji na systemie plików / For filesystem operations
import yaml  # Do wczytywania plików konfiguracyjnych / For loading configuration files
import shutil  # Do operacji na plikach wysokiego poziomu, np. usuwania folderów / For high-level file operations, e.g., removing directories
import time  # Do mierzenia czasu wykonania / For measuring execution time
import json  # Do pracy z plikami JSON (zapis logów) / For working with JSON files (log saving)
from datetime import datetime  # Do generowania znaczników czasu / For generating timestamps
# Biblioteki LangChain do integracji z Google Gemini
# LangChain libraries for Google Gemini integration
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# Biblioteka LangChain do przechowywania wektorów w ChromaDB
# LangChain library for storing vectors in ChromaDB
#from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
# Biblioteka LangChain do ładowania dokumentów z folderu
# LangChain library for loading documents from a directory
from langchain_community.document_loaders import DirectoryLoader
# Biblioteka LangChain do dzielenia tekstu na fragmenty (chunks)
# LangChain library for splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Biblioteki LangChain do budowania łańcucha RAG (prompt, przekazywanie danych, parser odpowiedzi)
# LangChain libraries for building the RAG chain (prompt, data passthrough, output parser)
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# --- Main Configuration Loading ---
# --- Główne ładowanie konfiguracji ---
def load_configuration():
    """
    Wczytuje konfigurację z pliku scripts/config.yaml i zwraca ją jako słownik.
    Loads configuration from scripts/config.yaml file and returns it as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'config.yaml')
    if not os.path.exists(config_path): config_path = "scripts/config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as cr:
            config = yaml.full_load(cr)
        conf = {
            "GOOGLE_API_KEY": config['KEY'],
            "BASE_PATH": config['base_path'],
            "TEXT_CACHE_FOLDER": os.path.join(config['base_path'], config['rag_pipeline_config']['text_cache_folder']),
            "VECTOR_DB_FOLDER": os.path.join(config['base_path'], config['rag_pipeline_config']['vector_db_folder']),
            "LOG_FOLDER": os.path.join(config['base_path'], "LOGS"),
            "EMBEDDING_MODEL": config['rag_pipeline_config']['embedding_model'],
            "LLM_MODEL": config['rag_pipeline_config']['llm_model']
        }
        print("Konfiguracja dla asystenta RAG załadowana pomyślnie.")
        print("RAG assistant configuration loaded successfully.")
        return conf
    except Exception as e:
        print(f"FATAL ERROR in configuration from path {config_path}: {e}")
        return None


# --- Log Management Function ---
# --- Funkcja Zarządzania Logami ---
def save_session_log(log_folder, log_data):
    """
    Zapisuje log z sesji zapytań i odpowiedzi do pliku JSON.
    Saves the query and answer session log to a JSON file.
    """
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_folder, f"rag_query_assistant_log_{session_timestamp}.json")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"\nLog sesji zapytań zapisano do: {log_file_path}")
        print(f"Query session log saved to: {log_file_path}")
    except IOError as e:
        print(f"BŁĄD: Nie można zapisać pliku logu sesji: {e}")
        print(f"ERROR: Could not write session log file: {e}")


# --- Initialization ---
# --- Inicjalizacja ---
CONFIG = load_configuration()
LLM = None
EMBEDDINGS = None
if CONFIG:
    try:
        LLM = GoogleGenerativeAI(model=CONFIG['LLM_MODEL'], google_api_key=CONFIG['GOOGLE_API_KEY'])
        EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=CONFIG['EMBEDDING_MODEL'],
                                                  google_api_key=CONFIG['GOOGLE_API_KEY'])
        os.makedirs(CONFIG['LOG_FOLDER'], exist_ok=True)
        print("Modele Gemini (LLM i Embeddings) gotowe do pracy.")
        print("Gemini model (LLM and Embeddings) are ready.")
    except Exception as e:
        print(f"BŁĄD: Nie można skonfigurować Gemini API: {e}")
        CONFIG = None


# --- Core RAG Functions ---
# --- Główne funkcje RAG ---
def build_vector_store_from_cache(session_log_ref):
    """
    Buduje (lub przebudowuje) bazę wektorową na podstawie plików .txt z folderu cache.
    Builds (or rebuilds) the vector store based on .txt files from the cache folder.
    """
    if not CONFIG: return

    start_time = time.time()
    log_entry = {"action": "build_vector_store", "start_utc": datetime.utcnow().isoformat(), "status": "In-Progress"}

    try:
        vector_db_path = CONFIG['VECTOR_DB_FOLDER']
        if os.path.exists(vector_db_path): shutil.rmtree(vector_db_path); print(
            f"Usunięto istniejącą bazę danych w: {vector_db_path}")

        loader = DirectoryLoader(CONFIG['TEXT_CACHE_FOLDER'], glob="**/*.txt", recursive=True)
        documents = loader.load()
        if not documents: raise FileNotFoundError(
            "Folder cache jest pusty. Uruchom najpierw 'rag_data_preprocessor.py'.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Budowanie bazy wektorowej z {len(docs)} fragmentów...");
        print(f"Building vector store from {len(docs)} chunks...")

        Chroma.from_documents(documents=docs, embedding=EMBEDDINGS, persist_directory=vector_db_path)
        log_entry["status"] = "Success"
        print(f"Baza wektorowa została pomyślnie zbudowana w: {vector_db_path}")

    except Exception as e:
        log_entry["status"] = "Failed";
        log_entry["error_message"] = str(e)
        print(f"KRYTYCZNY BŁĄD podczas budowania bazy wektorowej: {e}")

    log_entry["duration_seconds"] = round(time.time() - start_time, 2)
    session_log_ref["database_build_event"] = log_entry


def ask_assistant(query: str, session_log_ref):
    """
    Zadaje pytanie do istniejącej bazy wektorowej i zwraca odpowiedź.
    Asks a question to the existing vector store and returns the answer.
    """
    if not CONFIG: return

    start_time = time.time()
    log_entry = {"query": query}

    try:
        vector_db_path = CONFIG['VECTOR_DB_FOLDER']
        if not os.path.exists(vector_db_path): raise FileNotFoundError("Baza wektorowa nie istnieje.")

        vector_db = Chroma(persist_directory=vector_db_path, embedding_function=EMBEDDINGS)
        retriever = vector_db.as_retriever(search_kwargs={'k': 15})

        template = """Jesteś ekspertem prawnym specjalizującym się w analizie dokumentów. Odpowiedz na pytanie użytkownika precyzyjnie i wyłącznie na podstawie dostarczonego poniżej kontekstu. Jeśli w kontekście brakuje odpowiedzi, poinformuj o tym jasno. Cytuj kluczowe fakty.

KONTEKST Z TWOJEJ BAZY WIEDZY:
{context}

PYTANIE UŻYTKOWNIKA:
{question}

PRECYZYJNA ODPOWIEDŹ:"""
        prompt = PromptTemplate.from_template(template)

        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | LLM
                | StrOutputParser()
        )

        print("\n=======================================================")
        print(f"PYTANIE: {query}");
        print("---")

        response = rag_chain.invoke(query)
        print(f"ODPOWIEDŹ ASYSTENTA:\n{response}")

        log_entry["response"] = response;
        log_entry["status"] = "Success"

    except Exception as e:
        error_message = f"BŁĄD podczas przetwarzania zapytania: {e}";
        print(error_message)
        log_entry["response"] = error_message;
        log_entry["status"] = "Failed"

    log_entry["duration_seconds"] = round(time.time() - start_time, 2)
    session_log_ref["queries_asked"].append(log_entry)
    print("=======================================================\n")


# --- Main Execution ---
# --- Główne wykonanie ---
if __name__ == "__main__":
    """
    Główny blok wykonawczy skryptu. Pozwala na budowę bazy lub zadawanie pytań.
    Main execution block of the script. Allows building the database or asking questions.
    """
    session_log_data = {"session_start_utc": datetime.utcnow().isoformat(), "database_build_event": None,
                        "queries_asked": []}

    try:
        # === KROK 1: Budowanie bazy wektorowej (jednorazowo) ===
        # === STEP 1: Building the vector store (one-time) ===
        #build_vector_store_from_cache(session_log_data)

        # === KROK 2: Zadawanie pytań ===
        # === STEP 2: Asking questions ===
        if CONFIG and os.path.exists(CONFIG["VECTOR_DB_FOLDER"]):
            ask_assistant(
                "Czy posiadasz w bazie Konstytucję RP?",
                session_log_data)
            ask_assistant("Jesli tak - znajdź artykuł 13 i opisz mi jego znaczenie",
                          session_log_data)
        else:
            print(
                "\nBaza wektorowa nie istnieje. Uruchom najpierw `rag_data_preprocessor.py`, a następnie odkomentuj funkcję `build_vector_store_from_cache()` w tym skrypcie i uruchom go.")
            print(
                "Vector store does not exist. First run `rag_data_preprocessor.py`, then uncomment the `build_vector_store_from_cache()` function in this script and run it.")

    finally:
        # Zapisz log całej sesji na koniec
        # Save the entire session log at the end
        save_session_log(CONFIG['LOG_FOLDER'], session_log_data)