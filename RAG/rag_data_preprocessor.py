# --- Importing necessary libraries ---
# --- Importowanie niezbędnych bibliotek ---
import os  # Do operacji na systemie plików, np. tworzenia ścieżek / For filesystem operations, e.g., creating paths
import yaml  # Do wczytywania plików konfiguracyjnych w formacie YAML / For loading configuration files in YAML format
import time  # Do wstrzymywania wykonania skryptu (pauzy) / For pausing script execution (sleep)
import json  # Do pracy z plikami JSON (zapis i odczyt stanu) / For working with JSON files (state saving and loading)
import re  # Do operacji na wyrażeniach regularnych (czyszczenie nazw plików) / For regular expression operations (cleaning filenames)
from datetime import date, datetime, timedelta  # Do pracy z datą i czasem / For working with dates and time
import google.generativeai as genai  # Oficjalna biblioteka Google do API Gemini / Official Google library for the Gemini API
from PIL import Image  # Biblioteka Pillow do otwierania i manipulacji obrazami / Pillow library for opening and manipulating images
import pdfplumber  # Biblioteka do ekstrakcji tekstu i danych z plików PDF / Library for extracting text and data from PDF files


# --- Main Configuration Loading ---
# --- Główne ładowanie konfiguracji ---
def load_configuration():
    """
    Wczytuje konfigurację z pliku scripts/config.yaml i zwraca ją jako słownik.
    Loads configuration from scripts/config.yaml file and returns it as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'config.yaml')
    if not os.path.exists(config_path):
        config_path = "scripts/config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as cr:
            config = yaml.full_load(cr)
        conf = {
            "GOOGLE_API_KEY": config['KEY'], "BASE_PATH": config['base_path'],
            "SOURCE_FOLDER": os.path.join(config['base_path'], config['rag_pipeline_config']['source_folder']),
            "TEXT_CACHE_FOLDER": os.path.join(config['base_path'], config['rag_pipeline_config']['text_cache_folder']),
            "LOG_FOLDER": os.path.join(config['base_path'], "LOGS"),
            "LLM_MODEL": config['rag_pipeline_config']['llm_model'],
            "DAILY_LIMIT": config['rag_pipeline_config']['daily_api_request_limit'],
            "LIMIT_MARGIN": config['rag_pipeline_config']['limit_safety_margin']
        }
        print("Konfiguracja dla preprocesora RAG załadowana pomyślnie.")
        print("RAG preprocessor configuration loaded successfully.")
        return conf
    except Exception as e:
        print(f"FATAL ERROR in configuration from path {config_path}: {e}")
        return None


# --- State and Log Management Functions ---
# --- Funkcje Zarządzania Stanem i Logami ---
def load_state(state_file_path):
    today = date.today().isoformat()
    default_state = {"last_run_date": today, "requests_made_today": 0}
    if not os.path.exists(state_file_path):
        print("Plik stanu nie istnieje. Tworzę nowy stan na dzisiaj.")
        return default_state
    try:
        with open(state_file_path, 'r') as f:
            state = json.load(f)
        if state.get("last_run_date") != today:
            print("Wykryto nowy dzień. Resetuję licznik zapytań API.")
            return default_state
        return state
    except (json.JSONDecodeError, IOError):
        print("Błąd odczytu pliku stanu. Zaczynam od nowa.")
        return default_state


def save_state(state_file_path, state):
    try:
        with open(state_file_path, 'w') as f:
            json.dump(state, f, indent=4)
    except IOError as e:
        print(f"KRYTYCZNY BŁĄD: Nie można zapisać stanu do pliku '{state_file_path}': {e}")


def save_session_log(log_folder, log_data):
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_folder, f"rag_preprocessor_log_{session_timestamp}.json")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"Szczegółowy log sesji zapisano do: {log_file_path}")
    except IOError as e:
        print(f"BŁĄD: Nie można zapisać pliku logu sesji: {e}")


# --- Initialization ---
# --- Inicjalizacja ---
CONFIG = load_configuration()
MODEL = None
if CONFIG:
    try:
        genai.configure(api_key=CONFIG['GOOGLE_API_KEY'])
        MODEL = genai.GenerativeModel(CONFIG['LLM_MODEL'])
        os.makedirs(CONFIG['TEXT_CACHE_FOLDER'], exist_ok=True)
        os.makedirs(CONFIG['LOG_FOLDER'], exist_ok=True)
        print("Model Gemini i foldery gotowe do pracy.")
    except Exception as e:
        print(f"BŁĄD: Nie można skonfigurować Gemini API: {e}")
        CONFIG = None


# --- File Processing Functions ---
# --- Funkcje przetwarzania plików ---
def ocr_image_with_gemini(image_path_or_object):
    image_name = os.path.basename(image_path_or_object) if isinstance(image_path_or_object, str) else "PIL Image"
    print(f"  Przetwarzanie obrazu (OCR): {image_name}")
    try:
        img = Image.open(image_path_or_object) if isinstance(image_path_or_object, str) else image_path_or_object
        response = MODEL.generate_content(["Zrób OCR tego obrazu. Zwróć tylko i wyłącznie tekst.", img])
        return response.text, True
    except Exception as e:
        return f"[BŁĄD OCR: {e}]", False


def transcribe_audio_with_gemini(audio_path):
    print(f"  Przetwarzanie audio (Transkrypcja): {os.path.basename(audio_path)}")
    try:
        audio_file = genai.upload_file(path=audio_path)
        while audio_file.state.name == "PROCESSING":
            time.sleep(10)
            audio_file = genai.get_file(audio_file.name)
        if audio_file.state.name == "FAILED":
            raise ValueError(f"Przetwarzanie pliku w Gemini API nie powiodło się: {audio_file.name}")
        response = MODEL.generate_content(["Zrób transkrypcję tego nagrania. Zwróć tylko tekst.", audio_file])
        genai.delete_file(audio_file.name)
        return response.text, True
    except Exception as e:
        return f"[BŁĄD TRANSKRYPCJI: {e}]", False


def extract_text_from_pdf(pdf_path):
    print(f"  Przetwarzanie pliku PDF: {os.path.basename(pdf_path)}")
    full_text = ""
    api_calls_in_pdf = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        if not full_text.strip():
            print("  PDF wygląda na skan, uruchamiam OCR dla każdej strony...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if CONFIG and (load_state(os.path.join(CONFIG['BASE_PATH'], "RAG/preprocessor_state.json"))[
                                       'requests_made_today'] + api_calls_in_pdf >= CONFIG['DAILY_LIMIT'] - CONFIG[
                                       'LIMIT_MARGIN']):
                        print(f"    Zatrzymano przetwarzanie stron w PDF z powodu zbliżającego się limitu API.")
                        break
                    print(f"    OCR strony {page_num}/{len(pdf.pages)}...")
                    img = page.to_image(resolution=150).original
                    page_text, api_call_made = ocr_image_with_gemini(img)
                    if api_call_made:
                        api_calls_in_pdf += 1
                        time.sleep(2)
                    full_text += page_text + "\n\n--- Page Break ---\n\n"
        return full_text, api_calls_in_pdf
    except Exception as e:
        return f"[BŁĄD EKSTRAKCJI PDF: {e}]", 0


def read_text_file(txt_path):
    print(f"  Wczytywanie pliku tekstowego: {os.path.basename(txt_path)}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read(), 0
    except Exception as e:
        return f"[BŁĄD ODCZYTU TXT: {e}]", 0


# --- Main Pre-processing Logic ---
# --- Główna logika pre-processingu ---
def main():
    if not CONFIG:
        print("Zamykanie skryptu z powodu błędów konfiguracji.")
        return

    session_start_time = time.time()
    state_file = os.path.join(CONFIG['BASE_PATH'], "RAG/preprocessor_state.json")
    state = load_state(state_file)
    session_log = {"session_start_utc": datetime.utcnow().isoformat(),
                   "daily_api_requests_start": state['requests_made_today'], "processed_files_in_session": [],
                   "session_end_utc": None, "total_duration_seconds": None, "final_status": "Incomplete"}

    try:
        print("\n--- Rozpoczynam Pre-processing Danych dla RAG ---")
        limit_threshold = CONFIG['DAILY_LIMIT'] - CONFIG['LIMIT_MARGIN']
        supported_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp3', '.wav', '.m4a', '.flac',
                                '.ogg', '.txt')
        source_files_to_process = []
        for root, _, files in os.walk(CONFIG['SOURCE_FOLDER']):
            for file in files:
                if not file.lower().endswith(supported_extensions):
                    continue
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, CONFIG['SOURCE_FOLDER'])
                sanitized_cache_name = re.sub(r'[^a-zA-Z0-9._-]', '_', relative_path)
                cache_file_path = os.path.join(CONFIG['TEXT_CACHE_FOLDER'], f"{sanitized_cache_name}.txt")
                if not os.path.exists(cache_file_path):
                    source_files_to_process.append(source_path)

        if not source_files_to_process:
            print("\nWszystkie pliki źródłowe zostały już przetworzone.")
            session_log["final_status"] = "Completed - No new files"
            return

        print(f"Znaleziono {len(source_files_to_process)} nowych plików do przetworzenia.")

        for source_path in source_files_to_process:
            if state['requests_made_today'] >= limit_threshold:
                print("\nOsiągnięto próg bezpieczeństwa limitu API!")
                session_log["final_status"] = "Paused - API limit reached"
                break

            file_start_time = time.time()
            filename = os.path.basename(source_path)
            file_log_entry = {"filename": filename, "source_path": source_path, "status": "Processing", "api_calls": 0}
            print(f"\nPrzetwarzanie pliku: {filename}")

            extracted_text = ""
            api_calls_count = 0
            file_ext = os.path.splitext(filename)[1].lower()

            try:
                if file_ext == '.pdf':
                    extracted_text, api_calls_count = extract_text_from_pdf(source_path)
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    extracted_text, api_call_made = ocr_image_with_gemini(source_path)
                    api_calls_count = 1 if api_call_made else 0
                elif file_ext in ['.mp3', '.wav', '.m4a']:
                    extracted_text, api_call_made = transcribe_audio_with_gemini(source_path)
                    api_calls_count = 1 if api_call_made else 0
                elif file_ext == '.txt':
                    extracted_text, api_calls_count = read_text_file(source_path)

                if "BŁĄD" in extracted_text:
                    file_log_entry["status"] = "Error"
                else:
                    file_log_entry["status"] = "Success"
            except Exception as e:
                extracted_text = f"[BŁĄD KRYTYCZNY: {e}]"
                file_log_entry["status"] = "Critical Error"

            if api_calls_count > 0:
                state['requests_made_today'] += api_calls_count
                print(f"  Licznik API: {state['requests_made_today']}/{limit_threshold}")

            relative_path = os.path.relpath(source_path, CONFIG['SOURCE_FOLDER'])
            sanitized_cache_name = re.sub(r'[^a-zA-Z0-9._-]', '_', relative_path)
            cache_file_path = os.path.join(CONFIG['TEXT_CACHE_FOLDER'], f"{sanitized_cache_name}.txt")

            with open(cache_file_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"  Zapisano przetworzony tekst w cache: {os.path.basename(cache_file_path)}")

            file_log_entry["api_calls"] = api_calls_count
            file_log_entry["duration_seconds"] = round(time.time() - file_start_time, 2)
            session_log["processed_files_in_session"].append(file_log_entry)

        if session_log["final_status"] == "Incomplete":
            session_log["final_status"] = "Completed - All new files processed"

    finally:
        session_end_time = time.time()
        session_log["session_end_utc"] = datetime.utcnow().isoformat()
        session_log["total_duration_seconds"] = round(session_end_time - session_start_time, 2)
        print("\nZapisywanie stanu i logu sesji...")
        save_state(state_file, state)
        save_session_log(CONFIG['LOG_FOLDER'], session_log)
        print("--- Skrypt zakończył działanie ---")


# --- Ta linia musi być ostatnia w pliku ---
# --- This line must be the last one in the file ---
if __name__ == "__main__":
    main()