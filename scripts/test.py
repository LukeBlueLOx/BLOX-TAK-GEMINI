# --- Importing necessary libraries ---
# --- Importowanie niezbędnych bibliotek ---
# For filesystem operations like creating paths and folders. / Do operacji na systemie plików, jak tworzenie ścieżek i folderów.
import os
# For opening and extracting text from PDF files. / Do otwierania i wyciągania tekstu z plików PDF.
import pdfplumber
# For generating unique timestamps for filenames. / Do generowania unikalnych znaczników czasu dla nazw plików.
import datetime
# For loading configuration files in YAML format. / Do wczytywania plików konfiguracyjnych w formacie YAML.
import yaml
# For system interaction, e.g., to exit the script. / Do interakcji z systemem, np. do przerwania działania skryptu.
import sys
# For using regular expressions to fix line wrapping. / Do używania wyrażeń regularnych w celu naprawy zawijania wierszy.
import re
# For structured logging in JSON format. / Do strukturalnego logowania w formacie JSON.
import json
# For timing operations and pausing the script. / Do mierzenia czasu operacji i pauzowania skryptu.
import time
# The official Google library for interacting with the Gemini API. / Oficjalna biblioteka Google do interakcji z API Gemini.
import google.generativeai as genai
# For handling specific API errors like rate limiting. / Do obsługi specyficznych błędów API, takich jak limity zapytań.
from google.api_core import exceptions

# --- ReportLab imports for advanced PDF creation with paragraphs ---
# --- Importy ReportLab do zaawansowanego tworzenia PDF z akapitami ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_JUSTIFY


# --- MAIN CONFIGURATION LOADING FUNCTION ---
# --- GŁÓWNA FUNKCJA ŁADUJĄCA KONFIGURACJĘ ---
def load_configuration(config_path='config.yaml'):
    """
    Loads configuration from a YAML file.
    Wczytuje konfigurację z pliku YAML.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as cr:
            config = yaml.full_load(cr)

        if 'base_path' not in config:
            raise ValueError("Key 'base_path' is required.")
        if 'translator_script_config' not in config:
            raise ValueError("Section 'translator_script_config' is required.")
        if 'KEY' not in config or not config['KEY'] or config['KEY'] == "TWOJ_KLUCZ_API_GEMINI" or config[
            'KEY'] == "*****":
            raise ValueError("Gemini API Key ('KEY') is missing or is a placeholder.")

        base_path = config['base_path']
        translator_config = config['translator_script_config']

        conf = {
            "GEMINI_API_KEY": config['KEY'],
            "SOURCE_FOLDER": os.path.join(base_path, translator_config['source_folder']),
            "OUTPUT_FOLDER": os.path.join(base_path, translator_config['output_folder']),
            "LOG_FOLDER": os.path.join(base_path, translator_config.get('log_folder', 'LOGS')),
            "FONT_PATH": os.path.join(base_path, translator_config['font_path']),
            "FONT_NAME": translator_config['font_name'],
            "FONT_SIZE": translator_config.get('font_size', 10),
            "MODEL_NAME": translator_config.get('model_name', 'gemini-1.5-flash'),
            "CHUNK_SIZE_CHARS": translator_config.get('chunk_size_chars', 30000),
            "TARGET_LANGUAGES": translator_config.get('target_languages', ['English', 'Polish', 'Czech'])
        }
        print("Configuration loaded successfully.")
        print("Konfiguracja załadowana pomyślnie.")
        return conf
    except Exception as e:
        print(f"FATAL ERROR in configuration: {e}")
        print(f"BŁĄD KRYTYCZNY w konfiguracji: {e}")
        return None


# --- Initialization ---
# --- Inicjalizacja ---
CONFIG = load_configuration()
MODEL = None
if CONFIG:
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        genai.configure(api_key=CONFIG['GEMINI_API_KEY'])
        MODEL = genai.GenerativeModel(CONFIG['MODEL_NAME'], safety_settings=safety_settings)
        print("Gemini API configured successfully.")
        print("Gemini API skonfigurowane pomyślnie.")
    except Exception as e:
        CONFIG = None
    if CONFIG:
        os.makedirs(CONFIG['SOURCE_FOLDER'], exist_ok=True)
        os.makedirs(CONFIG['OUTPUT_FOLDER'], exist_ok=True)
        os.makedirs(CONFIG['LOG_FOLDER'], exist_ok=True)
        try:
            pdfmetrics.registerFont(TTFont(CONFIG['FONT_NAME'], CONFIG['FONT_PATH']))
            print(f"Font '{CONFIG['FONT_NAME']}' registered successfully.")
            print(f"Czcionka '{CONFIG['FONT_NAME']}' zarejestrowana pomyślnie.")
        except Exception as e:
            print(f"ERROR: Could not register font. Defaulting to Helvetica. Error: {e}")
            print(f"BŁĄD: Nie można zarejestrować czcionki. Używam domyślnej Helvetica. Błąd: {e}")
            CONFIG['FONT_NAME'] = "Helvetica"
else:
    print("Exiting script due to configuration errors.")
    print("Zamykanie skryptu z powodu błędów konfiguracji.")
    sys.exit(1)


# --- Core Functions ---
# --- Główne Funkcje ---

def extract_full_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file into a single string.
    Ekstrahuje cały tekst z pliku PDF do jednego ciągu znaków.
    """
    print(f"Extracting full text from '{os.path.basename(pdf_path)}'...")
    print(f"Ekstrahuję pełny tekst z '{os.path.basename(pdf_path)}'...")
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"ERROR: Could not read PDF file {pdf_path}: {e}")
        print(f"BŁĄD: Nie można odczytać pliku PDF {pdf_path}: {e}")
        return None


def chunk_text(text, chunk_size):
    """
    Splits a large text into smaller chunks based on a character size limit.
    Dzieli duży tekst na mniejsze kawałki na podstawie limitu znaków.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def translate_text_in_chunks(text_to_translate, target_language):
    """
    Translates text, handles rate limits and overloads by retrying, and manages other safety blocks.
    Tłumaczy tekst, obsługuje limity i przeciążenia poprzez ponawianie prób i zarządza innymi blokadami.
    """
    if not text_to_translate or not text_to_translate.strip():
        return "", 0, 0

    chunks = chunk_text(text_to_translate, CONFIG['CHUNK_SIZE_CHARS'])
    print(f"Text divided into {len(chunks)} chunk(s) for translation to {target_language}.")
    print(f"Tekst podzielony na {len(chunks)} części do tłumaczenia na {target_language}.")

    translated_parts = []
    total_input_tokens = 0
    total_output_tokens = 0

    base_prompt = f"You are a professional translator. Translate the following document fragment into {target_language}. Preserve the original formatting, including paragraph breaks. Return only the translated text, without any additional comments, explanations or introductions."

    for i, chunk in enumerate(chunks):
        print(f"  Translating chunk {i + 1}/{len(chunks)}...")
        print(f"  Tłumaczę część {i + 1}/{len(chunks)}...")
        prompt = f"{base_prompt}\n\nFragment to translate:\n\n{chunk}"

        max_retries = 5
        current_retry = 0
        delay = 15

        while current_retry < max_retries:
            try:
                response = MODEL.generate_content(prompt)

                if not response.parts:
                    block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"
                    print(
                        f"  WARNING: Chunk {i + 1} translation blocked by API. Reason: {block_reason}. Inserting original text.")
                    print(
                        f"  OSTRZEŻENIE: Tłumaczenie fragmentu {i + 1} zablokowane przez API. Powód: {block_reason}. Wstawiam oryginalny tekst.")
                    warning_msg = f"[API TRANSLATION BLOCKED (REASON: {block_reason}) - ORIGINAL TEXT INSERTED BELOW]"
                    translated_parts.append(f"\n\n--- {warning_msg} ---\n\n{chunk}\n\n")
                    break

                translated_parts.append(response.text)
                if hasattr(response, 'usage_metadata'):
                    total_input_tokens += response.usage_metadata.prompt_token_count
                    total_output_tokens += response.usage_metadata.candidates_token_count
                break

            except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.DeadlineExceeded) as e:
                current_retry += 1
                error_type = type(e).__name__
                if current_retry >= max_retries:
                    print(f"  ERROR: Max retries exceeded for chunk. Error: {error_type}. Inserting original text.")
                    print(
                        f"  BŁĄD: Przekroczono maksymalną liczbę prób dla fragmentu. Błąd: {error_type}. Wstawiam oryginalny tekst.")
                    translated_parts.append(
                        f"\n\n[TRANSLATION FAILED AFTER RETRIES: {error_type}] - ORIGINAL TEXT INSERTED BELOW\n\n{chunk}\n\n")
                    break

                print(
                    f"  API temporary error ({error_type}). Retrying in {delay} seconds... (Attempt {current_retry}/{max_retries})")
                print(
                    f"  Tymczasowy błąd API ({error_type}). Ponawiam próbę za {delay} sekund... (Próba {current_retry}/{max_retries})")
                time.sleep(delay)
                delay *= 2

            except Exception as e:
                print(f"  An unexpected API error occurred: {e}. Inserting original text.")
                print(f"  Wystąpił nieoczekiwany błąd API: {e}. Wstawiam oryginalny tekst.")
                translated_parts.append(
                    f"\n\n[UNEXPECTED TRANSLATION ERROR: {e}] - ORIGINAL TEXT INSERTED BELOW\n\n{chunk}\n\n")
                break

        time.sleep(1)

    print(f"Token Usage for {target_language}: Input={total_input_tokens}, Output={total_output_tokens}")
    print(f"Zużycie tokenów dla {target_language}: Wejście={total_input_tokens}, Wyjście={total_output_tokens}")
    return "\n\n".join(translated_parts), total_input_tokens, total_output_tokens


def reflow_text(text: str) -> str:
    """
    Intelligently joins lines for better PDF formatting.
    Inteligentnie łączy linie dla lepszego formatowania PDF.
    """
    reflowed = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    reflowed = re.sub(r'\n{2,}', '<br/><br/>', reflowed)
    return reflowed


def save_text_to_pdf(text_content, output_pdf_path):
    """
    Saves the given text content to a PDF file using Platypus for proper text wrapping.
    Zapisuje podany tekst do pliku PDF, używając biblioteki Platypus do poprawnego zawijania tekstu.
    """
    font_name, font_size = CONFIG['FONT_NAME'], CONFIG['FONT_SIZE']
    print(f"Saving PDF to: {output_pdf_path}...")
    print(f"Zapisuję PDF do: {output_pdf_path}...")

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    style = ParagraphStyle(name='Normal_Justified', parent=styles['Normal'], fontName=font_name, fontSize=font_size,
                           leading=font_size * 1.5, alignment=TA_JUSTIFY)

    processed_text = reflow_text(text_content)
    story = [Paragraph(processed_text, style)]

    try:
        doc.build(story)
        print(f"Successfully saved PDF: {output_pdf_path}")
        print(f"Pomyślnie zapisano PDF: {output_pdf_path}")
    except Exception as e:
        print(f"ERROR saving PDF: {e}")
        print(f"BŁĄD podczas zapisywania PDF: {e}")


# --- Main Execution ---
# --- Główne Wykonanie ---
def main():
    """
    Main function to find, translate, and save PDFs into separate files per language.
    Główna funkcja do znajdowania, tłumaczenia i zapisywania plików PDF w osobnych plikach dla każdego języka.
    """
    print(f"Starting PDF translation from folder: {CONFIG['SOURCE_FOLDER']}")
    print(f"Rozpoczynam tłumaczenie PDF z folderu: {CONFIG['SOURCE_FOLDER']}")

    pdf_files = sorted([f for f in os.listdir(CONFIG['SOURCE_FOLDER']) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print("INFO: No PDF files found in the source folder for translation.")
        print("INFO: Brak plików PDF w folderze źródłowym do tłumaczenia.")
        return

    for pdf_file in pdf_files:
        print(f"\n--- Processing file: {pdf_file} ---")
        print(f"--- Przetwarzam plik: {pdf_file} ---")

        pdf_path = os.path.join(CONFIG['SOURCE_FOLDER'], pdf_file)
        original_text = extract_full_text_from_pdf(pdf_path)

        if not original_text:
            print(f"WARNING: No text extracted from '{pdf_file}'. Skipping.")
            print(f"OSTRZEŻENIE: Nie wyekstrahowano tekstu z '{pdf_file}'. Pomijam.")
            continue

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(pdf_file)[0]

        # Mapping language names to file suffixes, sorted alphabetically by language name.
        # Mapowanie nazw języków na przyrostki plików, posortowane alfabetycznie według nazwy języka.
        lang_suffix_map = {
            "english": "EN",
            "polish": "PL",
            "czech": "CS",
            "ukrainian": "UA",
            "german": "DE",
            "french": "FR",
            "spanish": "ES",
            "russian": "RU",
            "arabic": "AR",
            "chinese": "ZH",
            "hebrew": "HE",
            "japanese": "JA",
            "persian": "FA",
        }

        for lang in CONFIG.get('TARGET_LANGUAGES', ['English', 'Polish']):
            print(f"\n--- Starting translation for {lang} ---")
            print(f"--- Rozpoczynam tłumaczenie na {lang} ---")

            translated_text, _, _ = translate_text_in_chunks(original_text, lang)

            if translated_text:
                lang_lower = lang.lower()
                lang_suffix = lang_suffix_map.get(lang_lower, lang_lower[:2].upper())
                output_path = os.path.join(CONFIG['OUTPUT_FOLDER'], f"{base_name}_{timestamp}_{lang_suffix}.pdf")

                save_text_to_pdf(translated_text, output_path)
            else:
                print(f"WARNING: Translation for {lang} resulted in empty text. No file will be saved.")
                print(f"OSTRZEŻENIE: Tłumaczenie na {lang} zwróciło pusty tekst. Plik nie zostanie zapisany.")

            sleep_duration = 20
            print(
                f"--- Finished {lang}. Waiting for {sleep_duration} seconds before next language to manage rate limits... ---")
            print(
                f"--- Ukończono {lang}. Czekam {sleep_duration} sekund przed kolejnym językiem w celu zarządzania limitami... ---")
            time.sleep(sleep_duration)

    print("\n--- Finished translation process for all files. ---")
    print("\n--- Zakończono proces tłumaczenia dla wszystkich plików. ---")


if __name__ == "__main__":
    if CONFIG:
        main()