# --- Importing necessary libraries ---
# --- Importowanie niezbędnych bibliotek ---
import os
import yaml
import shutil
import time
import json
from datetime import datetime

# LangChain libraries for Google Gemini integration
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ZAKTUALIZOWANY IMPORT: Używamy nowszej wersji biblioteki, aby uniknąć ostrzeżeń
from langchain_chroma import Chroma

# LangChain library for loading documents from a directory
from langchain_community.document_loaders import DirectoryLoader

# LangChain library for splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain libraries for building the RAG chain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# --- Main Configuration Loading ---
def load_configuration():
    """
    Loads configuration from scripts/config.yaml file and returns it as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'config.yaml')
    if not os.path.exists(config_path):
        config_path = "scripts/config.yaml"

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
def save_session_log(log_folder, log_data):
    """
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
def build_vector_store_from_cache(session_log_ref):
    """
    Builds (or rebuilds) the vector store based on .txt files from the cache folder.
    """
    if not CONFIG: return

    start_time = time.time()
    log_entry = {"action": "build_vector_store", "start_utc": datetime.utcnow().isoformat(), "status": "In-Progress"}

    try:
        vector_db_path = CONFIG['VECTOR_DB_FOLDER']
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
            print(f"Usunięto istniejącą bazę danych w: {vector_db_path}")

        loader = DirectoryLoader(CONFIG['TEXT_CACHE_FOLDER'], glob="**/*.txt", recursive=True)
        documents = loader.load()
        if not documents:
            raise FileNotFoundError("Folder cache jest pusty. Uruchom najpierw 'rag_data_preprocessor.py'.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Budowanie bazy wektorowej z {len(docs)} fragmentów...");
        print(f"Building vector store from {len(docs)} chunks...")

        # Zaktualizowana nazwa klasy Chroma
        Chroma.from_documents(documents=docs, embedding=EMBEDDINGS, persist_directory=vector_db_path)
        log_entry["status"] = "Success"
        print(f"Baza wektorowa została pomyślnie zbudowana w: {vector_db_path}")

    except Exception as e:
        log_entry["status"] = "Failed"
        log_entry["error_message"] = str(e)
        print(f"KRYTYCZNY BŁĄD podczas budowania bazy wektorowej: {e}")

    log_entry["duration_seconds"] = round(time.time() - start_time, 2)
    session_log_ref["database_build_event"] = log_entry


# ZMODYFIKOWANA FUNKCJA DO TESTÓW
def ask_assistant(query: str, session_log_ref):
    """
    Zadaje pytanie do istniejącej bazy wektorowej, POKAZUJE co znalazł retriever
    i zwraca odpowiedź.
    """
    if not CONFIG: return

    start_time = time.time()
    log_entry = {"query": query}

    try:
        vector_db_path = CONFIG['VECTOR_DB_FOLDER']
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError("Baza wektorowa nie istnieje.")

        # Zaktualizowana nazwa klasy Chroma
        vector_db = Chroma(persist_directory=vector_db_path, embedding_function=EMBEDDINGS)
        retriever = vector_db.as_retriever(search_kwargs={'k': 5})  # Szukamy 5 najbardziej pasujących fragmentów

        # === POCZĄTEK SEKCJI DIAGNOSTYCZNEJ ===
        # Ta sekcja sprawdzi, co retriever faktycznie znajduje w bazie, zanim przekaże to do LLM.
        print("\n" + "=" * 25 + " WYNIKI RETRIEVERA (DIAGNOSTYKA) " + "=" * 25)
        print(f"[DEBUG] Sprawdzanie, co retriever znajdzie dla zapytania: '{query}'")
        retrieved_docs = retriever.invoke(query)
        print(f"[DEBUG] Znaleziono {len(retrieved_docs)} dokumentów.")

        if not retrieved_docs:
            print("[DEBUG] UWAGA: Retriever nie zwrócił żadnych dokumentów. To jest główna przyczyna problemu!")
        else:
            for i, doc in enumerate(retrieved_docs):
                # 'source' jest standardowym polem w metadanych dodawanym przez DirectoryLoader
                source = doc.metadata.get('source', 'Brak metadanych o źródle')
                print(f"--- Fragment {i + 1} (ze źródła: {source}) ---")
                # Drukujemy pierwsze 400 znaków, aby nie zaśmiecać konsoli
                print(doc.page_content[:400].replace('\n', ' ') + "...")
                print("-" * 20)
        print("=" * 80 + "\n")
        # === KONIEC SEKCJI DIAGNOSTYCZNEJ ===

        # ZMODYFIKOWANY PROMPT: Bardziej neutralny, aby nie ograniczać modelu do jednej sprawy
        template = """Jesteś pomocnym asystentem AI. Odpowiedz na pytanie użytkownika precyzyjnie i wyłącznie na podstawie dostarczonego poniżej kontekstu. Jeśli w kontekście brakuje odpowiedzi, poinformuj o tym jasno.

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

        log_entry["response"] = response
        log_entry["status"] = "Success"

    except Exception as e:
        error_message = f"BŁĄD podczas przetwarzania zapytania: {e}"
        print(error_message)
        log_entry["response"] = error_message
        log_entry["status"] = "Failed"

    log_entry["duration_seconds"] = round(time.time() - start_time, 2)
    session_log_ref["queries_asked"].append(log_entry)
    print("=======================================================\n")


# --- Main Execution ---
if __name__ == "__main__":
    session_log_data = {
        "session_start_utc": datetime.utcnow().isoformat(),
        "database_build_event": None,
        "queries_asked": []
    }

    try:
        # === KROK 1: Budowanie bazy wektorowej (jednorazowo) ===
        # Jeśli chcesz przebudować bazę, odkomentuj poniższą linię.
        # W przeciwnym razie pozostaw ją zakomentowaną.
        # build_vector_store_from_cache(session_log_data)

        # === KROK 2: Zadawanie pytań ===
        if CONFIG and os.path.exists(CONFIG["VECTOR_DB_FOLDER"]):
            # Pytanie ogólne, które powinno znaleźć fragmenty z Konstytucji
            ask_assistant(
                "Czy posiadasz w bazie Konstytucję RP?",
                session_log_data
            )

            # Pytanie szczegółowe, które powinno znaleźć konkretny artykuł
            ask_assistant(
                "Znajdź artykuł 13 Konstytucji RP i opisz mi jego znaczenie",
                session_log_data
            )
        else:
            print(
                "\nBaza wektorowa nie istnieje. Uruchom najpierw `rag_data_preprocessor.py`, a następnie odkomentuj funkcję `build_vector_store_from_cache()` w tym skrypcie i uruchom go.")
            print(
                "Vector store does not exist. First run `rag_data_preprocessor.py`, then uncomment the `build_vector_store_from_cache()` function in this script and run it.")

    finally:
        # Zapisz log całej sesji na koniec
        if CONFIG:
            save_session_log(CONFIG['LOG_FOLDER'], session_log_data)