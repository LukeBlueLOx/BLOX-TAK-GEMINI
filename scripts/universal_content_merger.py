# For filesystem operations like creating paths and folders. / Do operacji na systemie plików, jak tworzenie ścieżek i folderów.
import os
# For opening and extracting text from PDF files, and converting pages to images. / Do otwierania i wyciągania tekstu z plików PDF oraz konwertowania stron na obrazy.
import pdfplumber
# For generating unique timestamps for filenames. / Do generowania unikalnych znaczników czasu dla nazw plików.
import datetime
# For loading configuration files in YAML format. / Do wczytywania plików konfiguracyjnych w formacie YAML.
import yaml
# For system interaction, e.g., to exit the script. / Do interakcji z systemem, np. do przerwania działania skryptu.
import sys
# For creating temporary files to handle image conversion for OCR. / Do tworzenia plików tymczasowych do obsługi konwersji obrazów dla OCR.
import tempfile
# For using regular expressions to find dates in filenames. / Do używania wyrażeń regularnych w celu znalezienia dat w nazwach plików.
import re
# For structured logging in JSON format. / Do strukturalnego logowania w formacie JSON.
import json
# For timing operations. / Do mierzenia czasu operacji.
import time
# The official Google library for interacting with the Gemini API. / Oficjalna biblioteka Google do interakcji z API Gemini.
import google.generativeai as genai

# ReportLab imports for advanced text wrapping and PDF creation
# Importy ReportLab do zaawansowanego zawijania tekstu i tworzenia PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT


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
            raise ValueError("Key 'base_path' is required in the configuration file.")
        if 'merger_script_config' not in config:
            raise ValueError("Section 'merger_script_config' is required in the configuration file.")
        if 'KEY' not in config or not config['KEY'] or config['KEY'] == "TWOJ_KLUCZ_API_GEMINI" or config[
            'KEY'] == "*****":
            raise ValueError("Gemini API Key ('KEY') not found, is empty, or is a placeholder.")

        base_path = config['base_path']
        merger_config = config['merger_script_config']

        conf = {
            "GEMINI_API_KEY": config['KEY'],
            "SOURCE_FOLDER": os.path.join(base_path, merger_config.get('source_folder', 'TEMP')),
            "OUTPUT_FOLDER": os.path.join(base_path, merger_config.get('output_folder', 'FOR_ANALYSIS')),
            "LOG_FOLDER": os.path.join(base_path, merger_config.get('log_folder', 'LOGS')),
            "FONT_PATH": os.path.join(base_path, merger_config.get('font_path', 'UbuntuMono-Regular.ttf')),
            "FONT_NAME": merger_config.get('font_name', 'UbuntuMono'),
            "FONT_SIZE": merger_config.get('font_size', 10),
            "MODEL_NAME": merger_config.get('model_name', 'gemini-1.5-flash'),
            "OCR_PROMPT": merger_config.get('ocr_prompt',
                                            'GEMINI, Make OCR. Do not add any additional information, just the text.'),
            "AUDIO_PROMPT": merger_config.get('audio_prompt',
                                              'Transcribe the audio recording. Return only the final text.'),
            "OCR_RESOLUTION": merger_config.get('ocr_resolution', 150)
        }

        print("Configuration loaded successfully.")
        print("Konfiguracja załadowana pomyślnie.")
        return conf

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"FATAL ERROR in configuration: {e}")
        print(f"BŁĄD KRYTYCZNY w konfiguracji: {e}")
        return None


# --- Initialization of Configuration and Services ---
# --- Inicjalizacja Konfiguracji i Usług ---
CONFIG = load_configuration()
MODEL = None

if CONFIG:
    try:
        genai.configure(api_key=CONFIG['GEMINI_API_KEY'])
        MODEL = genai.GenerativeModel(CONFIG['MODEL_NAME'])
        print(f"Gemini API configured successfully with model: {CONFIG['MODEL_NAME']}")
        print(f"Gemini API skonfigurowane pomyślnie z modelem: {CONFIG['MODEL_NAME']}")
    except Exception as e:
        print(f"ERROR: Could not configure Gemini API: {e}")
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
            print(
                f"ERROR: Could not register font from path '{CONFIG['FONT_PATH']}'. Defaulting to Helvetica. Error: {e}")
            CONFIG['FONT_NAME'] = "Helvetica"
else:
    print("Exiting script due to configuration errors.")
    sys.exit(1)

# --- File Type Constants ---
# --- Stałe typów plików ---
SUPPORTED_PDF_EXT = ('.pdf',)
SUPPORTED_TXT_EXT = ('.txt',)
SUPPORTED_IMG_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
SUPPORTED_AUDIO_EXT = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')


# --- Text Extraction Functions ---
# --- Funkcje Ekstrakcji Tekstu ---
def extract_text_from_text_pdf(pdf_path, start_page, end_page):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            actual_start_index = max(0, start_page - 1)
            actual_end_index = min(num_pages, end_page)

            if actual_start_index >= actual_end_index: return ""

            print(f"Extracting text from pages {actual_start_index + 1} to {actual_end_index}...")
            print(f"Ekstrakcja tekstu ze stron od {actual_start_index + 1} do {actual_end_index}...")
            for i in range(actual_start_index, actual_end_index):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"ERROR reading text-based PDF {pdf_path}: {e}")
        return f"[BŁĄD ODCZYTU PLIKU PDF: {e}]", None
    return text.strip(), {}


def extract_text_from_scanned_pdf(pdf_path, start_page, end_page):
    all_pages_text = []
    total_token_usage = {'prompt_token_count': 0, 'candidates_token_count': 0, 'total_token_count': 0}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            actual_start_index = max(0, start_page - 1)
            actual_end_index = min(num_pages, end_page)

            if actual_start_index >= actual_end_index: return "", {}

            print(
                f"PDF appears to be a scan. Performing OCR on pages {actual_start_index + 1} to {actual_end_index}...")
            print(f"PDF wygląda na skan. Wykonuję OCR na stronach od {actual_start_index + 1} do {actual_end_index}...")

            for i in range(actual_start_index, actual_end_index):
                page = pdf.pages[i]
                img = page.to_image(resolution=CONFIG.get("OCR_RESOLUTION", 150))

                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image:
                    img.save(temp_image.name, format="PNG")
                    page_text, usage = extract_text_from_image_with_gemini(temp_image.name)
                    all_pages_text.append(page_text)
                    if usage:
                        total_token_usage['prompt_token_count'] += usage.get('prompt_token_count', 0)
                        total_token_usage['candidates_token_count'] += usage.get('candidates_token_count', 0)
                        total_token_usage['total_token_count'] += usage.get('total_token_count', 0)
    except Exception as e:
        print(f"ERROR during scanned PDF processing {pdf_path}: {e}")
        return f"[BŁĄD PRZETWARZANIA ZESKANOWANEGO PDF: {e}]", None

    return "\n\n--- Page Break ---\n\n".join(all_pages_text), total_token_usage


def extract_text_from_txt(txt_path):
    print(f"Reading text file: {os.path.basename(txt_path)}...")
    print(f"Odczytuję plik tekstowy: {os.path.basename(txt_path)}...")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read(), {}
    except Exception as e:
        print(f"ERROR reading TXT {txt_path}: {e}")
        return f"[BŁĄD ODCZYTU PLIKU TXT: {e}]", None


def extract_text_from_image_with_gemini(image_path):
    if not MODEL: return "[OCR ERROR: Gemini API is not configured]", None

    print(f"Performing OCR on image: {os.path.basename(image_path)}...")
    print(f"Przeprowadzam OCR na obrazie: {os.path.basename(image_path)}...")
    try:
        image_file = genai.upload_file(path=image_path)
        response = MODEL.generate_content([CONFIG['OCR_PROMPT'], image_file])
        genai.delete_file(image_file.name)
        usage_metadata = getattr(response, 'usage_metadata', None)
        usage_dict = {}
        if usage_metadata:
            usage_dict = {
                'prompt_token_count': usage_metadata.prompt_token_count,
                'candidates_token_count': usage_metadata.candidates_token_count,
                'total_token_count': usage_metadata.total_token_count
            }
        return response.text.strip(), usage_dict
    except Exception as e:
        print(f"ERROR during OCR on {image_path}: {e}")
        return f"[BŁĄD OCR GEMINI: {e}]", None


def extract_text_from_audio_with_gemini(audio_path):
    if not MODEL: return "[TRANSCRIPTION ERROR: Gemini API is not configured]", None

    print(f"Transcribing audio file: {os.path.basename(audio_path)}...")
    print(f"Przeprowadzam transkrypcję pliku audio: {os.path.basename(audio_path)}...")
    audio_file = None
    try:
        audio_file = genai.upload_file(path=audio_path)
        response = MODEL.generate_content([CONFIG['AUDIO_PROMPT'], audio_file])
        usage_metadata = getattr(response, 'usage_metadata', None)
        usage_dict = {}
        if usage_metadata:
            usage_dict = {
                'prompt_token_count': usage_metadata.prompt_token_count,
                'candidates_token_count': usage_metadata.candidates_token_count,
                'total_token_count': usage_metadata.total_token_count
            }
        return response.text.strip(), usage_dict
    except Exception as e:
        print(f"ERROR during audio transcription of {audio_path}: {e}")
        return f"[BŁĄD TRANSKRYPCJI AUDIO: {e}]", None
    finally:
        if audio_file:
            genai.delete_file(audio_file.name)
            print(f"Cleaned up temporary audio file: {audio_file.name}")
            print(f"Posprzątano tymczasowy plik audio: {audio_file.name}")


# --- Helper Functions and Main Logic ---
# --- Funkcje pomocnicze i główna logika ---
def extract_date_from_path(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', filename)
    if match:
        date_str = match.group(1).replace('-', '').replace('_', '')
        try:
            return datetime.datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            pass
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    except OSError:
        return datetime.datetime.min


def get_page_range_input(pdf_file_name, total_pages):
    while True:
        choice = input(f"For file '{pdf_file_name}' (total pages: {total_pages}):\n"
                       f"  1. Process all pages\n"
                       f"  2. Specify page range (e.g., 10-20)\n"
                       f"Dla pliku '{pdf_file_name}' (łącznie stron: {total_pages}):\n"
                       f"  1. Przetwórz wszystkie strony\n"
                       f"  2. Podaj zakres stron (np. 10-20)\n"
                       f"Choose an option (1/2) / Wybierz opcję (1/2): ").strip()
        if choice == '1':
            return 1, total_pages
        elif choice == '2':
            page_range_str = input("Enter page range / Podaj zakres stron: ").strip()
            try:
                start, end = map(int, page_range_str.split('-'))
                if 1 <= start <= end <= total_pages:
                    return start, end
                else:
                    print("ERROR: Invalid page range.")
            except ValueError:
                print("ERROR: Invalid format. Use 'START-END'.")
        else:
            print("Invalid choice.")


# --- NEW HELPER FUNCTION TO FIX WRAPPING ---
# --- NOWA FUNKCJA POMOCNICZA DO NAPRAWY ZAWIJANIA ---
def reflow_text(text: str) -> str:
    """
    Intelligently joins lines that have been incorrectly broken mid-sentence.
    Inteligentnie łączy linie, które zostały nieprawidłowo złamane w połowie zdania.
    """
    # Step 1: Replace single newlines (likely incorrect breaks) with a space. / Krok 1: Zamień pojedyncze znaki nowej linii (prawdopodobnie nieprawidłowe złamania) na spację.
    reflowed = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Step 2: Replace two or more newlines (which signify a paragraph break) with a <br/> tag understood by ReportLab. / Krok 2: Zamień dwa (lub więcej) znaki nowej linii (które oznaczają koniec akapitu) na tag <br/> zrozumiały dla ReportLab.
    reflowed = re.sub(r'\n{2,}', '<br/><br/>', reflowed)

    return reflowed


# --- MODIFIED PDF SAVING FUNCTION ---
# --- ZMODYFIKOWANA FUNKCJA ZAPISU DO PDF ---
def save_text_to_pdf(text_content, output_pdf_path):
    """
    Saves the given text content to a PDF file with proper text wrapping.
    Zapisuje podany tekst do pliku PDF z poprawnym zawijaniem tekstu.
    """
    font_name = CONFIG['FONT_NAME']
    font_size = CONFIG.get('FONT_SIZE', 10)
    print(f"Saving PDF with proper wrapping to: {output_pdf_path}...")
    print(f"Zapisuję PDF z poprawnym zawijaniem do: {output_pdf_path}...")

    # Setup document template. / Ustawienie szablonu dokumentu.
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    # Setup paragraph styles. / Ustawienie stylów akapitu.
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name='Normal_Justified',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.5,
        alignment=TA_JUSTIFY,  # Justify text for a cleaner look. / Justowanie tekstu dla czystszego wyglądu.
    )

    # *** KEY CHANGE: Call the `reflow_text` function before creating the Paragraph. ***
    # *** KLUCZOWA ZMIANA: Wywołanie funkcji `reflow_text` przed utworzeniem akapitu. ***
    processed_text = reflow_text(text_content)

    story = [Paragraph(processed_text, style)]

    # Build the PDF. / Zbuduj (wygeneruj) PDF.
    try:
        doc.build(story)
        print(f"Successfully saved PDF: {output_pdf_path}")
        print(f"Pomyślnie zapisano PDF: {output_pdf_path}")
    except Exception as e:
        print(f"ERROR saving PDF: {e}")
        print(f"BŁĄD podczas zapisywania PDF: {e}")


def write_summary_log(log_data):
    """
    Writes a JSON summary log for the entire session.
    Zapisuje podsumowujący log JSON dla całej sesji.
    """
    log_file_path = os.path.join(CONFIG['LOG_FOLDER'], f"merger_log_{log_data['session_start_iso']}.json")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print("\n--- Summary log created successfully. ---")
        print(f"--- Podsumowujący log został pomyślnie utworzony: {log_file_path} ---")
    except Exception as e:
        print(f"ERROR: Could not write summary log file: {e}")
        print(f"BŁĄD: Nie można zapisać pliku z logami: {e}")


def main():
    """
    Main function to merge content from PDF, TXT, image and audio files.
    Główna funkcja do łączenia treści z plików PDF, TXT, obrazów i audio.
    """
    script_start_time = time.time()
    session_start_iso = datetime.datetime.now().isoformat().replace(":", "-")

    print(f"Starting content merge from folder: {CONFIG['SOURCE_FOLDER']}")
    print(f"Rozpoczynam łączenie treści z folderu: {CONFIG['SOURCE_FOLDER']}")

    all_files = []
    supported_extensions = SUPPORTED_PDF_EXT + SUPPORTED_TXT_EXT + SUPPORTED_IMG_EXT + SUPPORTED_AUDIO_EXT
    for dirpath, _, filenames in os.walk(CONFIG['SOURCE_FOLDER']):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                all_files.append(os.path.join(dirpath, filename))

    print(f"Sorting {len(all_files)} files by date (newest first)...")
    print(f"Sortuję {len(all_files)} plików według daty (od najnowszych)...")
    all_files.sort(key=extract_date_from_path, reverse=True)

    if not all_files:
        print("INFO: No supported files found for processing.")
        print("INFO: Nie znaleziono obsługiwanych plików do przetworzenia.")
        return

    combined_text_parts = []
    session_log_details = []

    for file_path in all_files:
        file_start_time = time.time()
        file_name = os.path.basename(file_path)

        file_log = {"file_name": file_name, "full_path": file_path, "status": "Processing", "details": ""}
        print(f"\n--- Processing file: {file_name} ---")
        print(f"\n--- Przetwarzam plik: {file_name} ---")

        combined_text_parts.append(f"\n\n--- BEGINNING OF FILE: {file_name} ---\n\n")
        combined_text_parts.append(f"--- POCZĄTEK PLIKU: {file_name} ---\n\n")

        extracted_text = ""
        token_usage = {}
        error_message = None

        try:
            if file_name.lower().endswith(SUPPORTED_PDF_EXT):
                with pdfplumber.open(file_path) as pdf:
                    num_pages = len(pdf.pages)
                start_p, end_p = get_page_range_input(file_name, num_pages)
                extracted_text, token_usage = extract_text_from_text_pdf(file_path, start_p, end_p)
                if not extracted_text.strip():
                    extracted_text, token_usage = extract_text_from_scanned_pdf(file_path, start_p, end_p)
            elif file_name.lower().endswith(SUPPORTED_TXT_EXT):
                extracted_text, token_usage = extract_text_from_txt(file_path)
            elif file_name.lower().endswith(SUPPORTED_IMG_EXT):
                extracted_text, token_usage = extract_text_from_image_with_gemini(file_path)
            elif file_name.lower().endswith(SUPPORTED_AUDIO_EXT):
                extracted_text, token_usage = extract_text_from_audio_with_gemini(file_path)

            file_log['status'] = "Success"
        except Exception as e:
            error_message = str(e)
            extracted_text = f"[BŁĄD PRZETWARZANIA PLIKU: {error_message}]"
            file_log['status'] = "ERROR"
            file_log['details'] = error_message

        combined_text_parts.append(extracted_text)
        combined_text_parts.append(f"\n\n--- END OF FILE: {file_name} ---\n\n")
        combined_text_parts.append(f"--- KONIEC PLIKU: {file_name} ---\n\n")

        file_log['processing_time_seconds'] = round(time.time() - file_start_time, 2)
        if token_usage:
            file_log['token_usage'] = {
                'prompt': token_usage.get('prompt_token_count', 0),
                'candidates': token_usage.get('candidates_token_count', 0),
                'total': token_usage.get('total_token_count', 0)
            }
        session_log_details.append(file_log)

    script_end_time = time.time()
    if combined_text_parts:
        final_text = "".join(combined_text_parts)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf_path = os.path.join(CONFIG['OUTPUT_FOLDER'], f"Combined_Content_{timestamp}.pdf")

        # Calling the modified save function. / Wywołanie zmodyfikowanej funkcji zapisu.
        save_text_to_pdf(final_text, output_pdf_path)

        print("\n--- Finished merging all content. ---")
        print("\n--- Zakończono łączenie wszystkich treści. ---")
    else:
        print("\nNo text was extracted from any file.")
        print("\nNie wyekstrahowano tekstu z żadnego pliku.")

    overall_log = {
        "session_start_iso": session_start_iso,
        "total_duration_seconds": round(script_end_time - script_start_time, 2),
        "total_files_processed": len(session_log_details),
        "processed_files": session_log_details
    }
    write_summary_log(overall_log)


if __name__ == "__main__":
    if CONFIG:
        main()