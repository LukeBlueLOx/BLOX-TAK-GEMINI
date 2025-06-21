import os
import google.generativeai as genai
import datetime
import time
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import yaml

with open("config.yaml", "r") as cr:
    config_vals = yaml.full_load(cr)
KEY = config_vals['KEY']

# --- PATH CONFIGURATION ---
# --- KONFIGURACJA ŚCIEŻEK ---
# Folder for input images.
# Folder na obrazy wejściowe.
IMAGE_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "image_input")
# Folder for output analysis results.
# Folder na wyniki analizy.
IMAGE_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "image_output")
# Folder for logs of image processing.
# Folder na logi przetwarzania obrazów.
IMAGE_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "image_logs")

# Define the path to the Ubuntu Mono font file.
# Make sure 'UbuntuMono-R.ttf' is in the same directory as this script, or provide a full path.
# Zdefiniuj ścieżkę do pliku czcionki Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-R.ttf' znajduje się w tym samym katalogu co ten skrypt, lub podaj pełną ścieżkę.
UBUNTU_MONO_FONT_PATH = os.path.join(os.path.dirname(__file__), "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf")

# Create directories if they don't exist.
# Tworzy katalogi, jeśli nie istnieją.
os.makedirs(IMAGE_INPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_LOG_FOLDER, exist_ok=True)

# Register the Ubuntu Mono font with ReportLab.
# Zarejestruj czcionkę Ubuntu Mono w ReportLab.
try:
    pdfmetrics.registerFont(TTFont('UbuntuMono', UBUNTU_MONO_FONT_PATH))
    print(f"Font 'UbuntuMono' registered successfully from {UBUNTU_MONO_FONT_PATH}.")
    print(f"Czcionka 'UbuntuMono' zarejestrowana pomyślnie z {UBUNTU_MONO_FONT_PATH}.")
except Exception as e:
    print(
        f"ERROR: Could not register Ubuntu Mono font from {UBUNTU_MONO_FONT_PATH}. Please ensure the file exists and is accessible. Error: {e}")
    print(
        f"BŁĄD: Nie można zarejestrować czcionki Ubuntu Mono z {UBUNTU_MONO_FONT_PATH}. Upewnij się, że plik istnieje i jest dostępny. Błąd: {e}")
    # Fallback to a default font if Ubuntu Mono cannot be registered.
    # W razie problemów z rejestracją, użyj domyślnej czcionki.
    DEFAULT_FONT = "Helvetica"
else:
    DEFAULT_FONT = "UbuntuMono"

FONT_SIZE = 12  # Font size for the PDF output / Rozmiar czcionki dla wyjścia PDF

# --- GEMINI API KEY CONFIGURATION ---
# --- KONFIGURACJA KLUCZA API GEMINI ---
# IMPORTANT: Replace "*****" with your real API key in config.yaml!
# WAŻNE: Zastąp "*****" swoim prawdziwym kluczem API w pliku config.yaml!
GEMINI_API_KEY = KEY
# Configure the Gemini API with the provided key.
# Konfiguruje API Gemini za pomocą podanego klucza.
genai.configure(api_key=GEMINI_API_KEY)
print("Gemini API configured successfully.")
print("Gemini API skonfigurowane pomyślnie.")


# --- Functions for Image File Handling and Gemini Vision ---
# --- Funkcje do obsługi plików graficznych i Gemini Vision ---

def upload_image_to_gemini_files_api(image_file_path):
    # Print message indicating file upload.
    # Wyświetla komunikat o przesyłaniu pliku.
    print(f"Uploading image file '{os.path.basename(image_file_path)}' to Gemini Files API...")
    print(f"Przesyłanie pliku graficznego '{os.path.basename(image_file_path)}' do Gemini Files API...")
    try:
        # Upload the file using genai.upload_file.
        # Przesyła plik za pomocą genai.upload_file.
        file = genai.upload_file(path=image_file_path)
        print(f"File '{os.path.basename(image_file_path)}' uploaded. Gemini File Name: {file.name}")
        print(f"Plik '{os.path.basename(image_file_path)}' przesłany. Nazwa pliku Gemini: {file.name}")
        return file
    except Exception as e:
        # Error message if upload fails.
        # Komunikat o błędzie, jeśli przesłanie pliku się nie powiedzie.
        print(f"ERROR: Could not upload file '{image_file_path}' to Gemini Files API: {e}")
        print(f"BŁĄD: Nie można przesłać pliku '{image_file_path}' do Gemini Files API: {e}")
        return None


def analyze_image_with_gemini_model(image_file_object, model_name, prompt, log_data):
    # Print message indicating image analysis.
    # Wyświetla komunikat o analizie obrazu.
    print(f"Analyzing image with Gemini model '{model_name}'...")
    print(f"Analiza obrazu za pomocą modelu Gemini '{model_name}'...")

    # Initialize the GenerativeModel.
    # Inicjuje model GenerativeModel.
    model = genai.GenerativeModel(model_name=model_name)

    # Count tokens for input (image file and prompt).
    # Liczy tokeny dla inputu (pliku graficznego i promptu).
    try:
        # Note: For images, 'count_tokens' accepts a list of objects, including the image_file_object.
        # Uwaga: Dla obrazów 'count_tokens' przyjmuje listę obiektów, w tym image_file_object.
        count_response = model.count_tokens([image_file_object, prompt])
        input_token_count = count_response.total_tokens
        print(f"Input token count: {input_token_count}")
        print(f"Liczba tokenów wejściowych: {input_token_count}")
        log_data["input_tokens"] = input_token_count
    except Exception as e:
        # Warning if input tokens cannot be counted.
        # Ostrzeżenie, jeśli nie można policzyć tokenów wejściowych.
        print(f"WARNING: Could not count input tokens: {e}")
        print(f"OSTRZEŻENIE: Nie można policzyć tokenów wejściowych: {e}")
        log_data["input_tokens"] = "ERROR"

    response_text = None
    try:
        # Pass the image and prompt to generate_content.
        # Przekazuje obraz i prompt do generate_content.
        response = model.generate_content([image_file_object, prompt])
        response_text = response.text.strip()
        # Get output and total token counts from usage metadata.
        # Pobiera liczbę tokenów wyjściowych i całkowitą z metadanych użycia.
        log_data["output_tokens"] = response.usage_metadata.candidates_token_count if hasattr(response.usage_metadata,
                                                                                              'candidates_token_count') else "N/A"
        log_data["total_tokens"] = response.usage_metadata.total_token_count if hasattr(response.usage_metadata,
                                                                                        'total_token_count') else "N/A"
        print(f"Output token count: {log_data['output_tokens']}")
        print(f"Liczba tokenów wyjściowych: {log_data['output_tokens']}")
        print(f"Total tokens (input+output): {log_data['total_tokens']}")
        print(f"Łączna liczba tokenów (wejście+wyjście): {log_data['total_tokens']}")
        return response_text
    except Exception as e:
        # Error message if image analysis fails.
        # Komunikat o błędzie, jeśli analiza obrazu się nie powiedzie.
        print(f"ERROR: Could not analyze image with Gemini: {e}")
        print(f"BŁĄD: Nie można analizować obrazu z Gemini: {e}")
        log_data["error"] = str(e)
        return None


def delete_gemini_file(file_object):
    # Only proceed if file_object exists.
    # Kontynuuje tylko jeśli istnieje file_object.
    if file_object:
        # Print message indicating file deletion.
        # Wyświetla komunikat o usuwaniu pliku.
        print(f"Deleting file '{file_object.name}' from Gemini Files API...")
        print(f"Usuwanie pliku '{file_object.name}' z Gemini Files API...")
        try:
            # Delete the file from Gemini Files API.
            # Usuwa plik z Gemini Files API.
            genai.delete_file(file_object.name)
            print("File deleted successfully.")
            print("Plik usunięty pomyślnie.")
        except Exception as e:
            # Error message if deletion fails.
            # Komunikat o błędzie, jeśli usunięcie się nie powiedzie.
            print(f"ERROR: Could not delete file '{file_object.name}' from Gemini Files API: {e}")
            print(f"BŁĄD: Nie można usunąć pliku '{file_object.name}' z Gemini Files API: {e}")


def get_image_analyzer_function(analyzer_type="gemini_native"):
    # Check if the analyzer type is 'gemini_native'.
    # Sprawdza, czy typ analizatora to 'gemini_native'.
    if analyzer_type == "gemini_native":
        # Model for image analysis (multimodal).
        # Model do analizy obrazów (multimodalny).
        # Use "gemini-1.5-flash" or "gemini-1.5-pro" for more advanced analysis.
        # Użyj "gemini-1.5-flash" lub "gemini-1.5-pro" dla bardziej zaawansowanych analiz.
        supported_image_model = "gemini-1.5-flash"

        print(f"Image analysis model set to: {supported_image_model}")
        print(f"Ustawiono model do analizy obrazów na: {supported_image_model}")

        # Define the inner function for Gemini native image analysis.
        # Definiuje wewnętrzną funkcję do natywnej analizy obrazów Gemini.
        def _analyze_image_with_gemini_native(image_file_path, prompt_for_image, log_data):
            # Upload the image to Gemini Files API.
            # Przesyła obraz do Gemini Files API.
            file_obj = upload_image_to_gemini_files_api(image_file_path)
            if file_obj:
                log_data["gemini_file_name"] = file_obj.name
                try:
                    # Analyze the image using the Gemini model.
                    # Analizuje obraz za pomocą modelu Gemini.
                    analysis_text = analyze_image_with_gemini_model(file_obj,
                                                                    model_name=supported_image_model,
                                                                    prompt=prompt_for_image,
                                                                    log_data=log_data)
                    return analysis_text
                finally:
                    # Ensure the file is deleted from Gemini Files API after analysis.
                    # Upewnia się, że plik zostanie usunięty z Gemini Files API po analizie.
                    delete_gemini_file(file_obj)
            return None

        return _analyze_image_with_gemini_native
    else:
        # Raise an error for an unknown analyzer type.
        # Podnosi błąd dla nieznanego typu analizatora.
        raise ValueError(f"Unknown image analysis type: {analyzer_type}. Please choose 'gemini_native'.")
        raise ValueError(f"Nieznany typ analizy obrazu: {analyzer_type}. Proszę wybrać 'gemini_native'.")


# --- New function to save text to PDF ---
# --- Nowa funkcja do zapisywania tekstu do PDF ---
def save_text_to_pdf(text_content, output_pdf_path, font_name, font_size):
    print(f"Saving text to PDF: {output_pdf_path} with font '{font_name}' {font_size}pt...")
    print(f"Zapisywanie tekstu do PDF: {output_pdf_path} czcionką '{font_name}' {font_size}pkt...")
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        c.setFont(font_name, font_size)
        width, height = A4

        # Margins / Marginesy
        left_margin = 50
        top_margin = height - 50
        line_height = font_size * 1.2  # Adjust for spacing / Dostosuj do odstępów

        y_position = top_margin

        # Split text into lines, handling long lines
        # Podziel tekst na linie, obsługując długie linie
        lines = []
        for paragraph in text_content.split('\n'):
            wrapped_lines = []
            if paragraph.strip():  # Avoid processing empty paragraphs
                # Calculate max characters per line for the given font and size
                # Oblicz maksymalną liczbę znaków na linię dla danej czcionki i rozmiaru
                char_width = pdfmetrics.stringWidth('M', font_name, font_size)  # Width of a typical character
                max_chars_per_line = int((width - 2 * left_margin) / char_width) if char_width > 0 else 100

                # ReportLab's textobject handles wrapping better for monospaced fonts,
                # but a manual wrap ensures we control chunking.
                # Obiekt tekstowy ReportLab lepiej obsługuje zawijanie dla czcionek monospaced,
                # ale ręczne zawijanie zapewnia kontrolę nad chunkowaniem.
                current_line = ""
                for word in paragraph.split(' '):
                    if pdfmetrics.stringWidth(current_line + (word + ' '), font_name, font_size) < (
                            width - 2 * left_margin):
                        current_line += (word + ' ')
                    else:
                        wrapped_lines.append(current_line.strip())
                        current_line = word + ' '
                if current_line.strip():
                    wrapped_lines.append(current_line.strip())
            else:
                wrapped_lines.append("")  # Keep empty lines for paragraph breaks
            lines.extend(wrapped_lines)

        for line in lines:
            if y_position < 50:  # Check if new page is needed (50 is bottom margin)
                c.showPage()
                c.setFont(font_name, font_size)
                y_position = top_margin

            c.drawString(left_margin, y_position, line)
            y_position -= line_height

        c.save()
        print(f"Successfully saved PDF to: {output_pdf_path}")
        print(f"Pomyślnie zapisano PDF do: {output_pdf_path}")
    except Exception as e:
        print(f"ERROR: Could not save text to PDF '{output_pdf_path}': {e}")
        print(f"BŁĄD: Nie można zapisać tekstu do PDF '{output_pdf_path}': {e}")


def process_image_folder():
    print(f"\n--- Starting scan and processing of image files from folder: {IMAGE_INPUT_FOLDER} ---")
    print(f"\n--- Rozpoczynam skanowanie i przetwarzanie plików graficznych z folderu: {IMAGE_INPUT_FOLDER} ---")

    # Supported image file extensions.
    # Obsługiwane rozszerzenia plików graficznych.
    image_files = [f for f in os.listdir(IMAGE_INPUT_FOLDER)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".pdf"))]

    if not image_files:
        print("INFO: No image files found in the folder for analysis.")
        print("INFO: Brak plików graficznych w folderze do analizy.")
        return

    # You can define a single prompt for all images or customize it for each.
    # Możesz zdefiniować pojedynczy prompt dla wszystkich obrazów lub dostosować go dla każdego z osobna.
    # Example of a simple prompt:
    # Przykład prostego promptu:
    default_image_prompt = "GEMINI, Make OCR, And Make Output in Polish and English Language. Do not add any additional information, just the text."  # "Describe in detail what you see in the image. Indicate the most important elements, colors, text (if present), and the general mood or context."
    # default_image_prompt = "Opisz szczegółowo co widzisz na obrazie. Wskaż najważniejsze elementy, kolory, tekst (jeśli występuje) oraz ogólny nastrój lub kontekst."

    # Example of a more advanced prompt (if you want to analyze documents, for example):
    # Przykład bardziej zaawansowanego promptu (jeśli chcesz analizować np. dokumenty):
    # default_image_prompt = "Transcribe all visible text in the image. Then, summarize the main content of the document and indicate if it contains any dates or key data."
    # default_image_prompt = "Przetranskrybuj cały tekst widoczny na obrazie. Następnie, podsumuj główną treść dokumentu i wskaż, czy zawiera jakiekolwiek daty lub kluczowe dane."

    # Get the image analyzer function.
    # Pobiera funkcję analizującą obrazy.
    analyze_image_function = get_image_analyzer_function("gemini_native")

    total_files_processed = 0
    total_tokens_used = 0
    total_processing_time = 0.0

    # Global list to collect logs for all files.
    # Globalna lista do zbierania logów dla wszystkich plików.
    overall_logs = []

    for image_file in image_files:
        image_path = os.path.join(IMAGE_INPUT_FOLDER, image_file)
        base_name = os.path.splitext(image_file)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.txt")
        output_pdf_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.pdf")  # New PDF path
        log_file_path = os.path.join(IMAGE_LOG_FOLDER, f"{base_name}_log_{timestamp}.json")

        # Dictionary for logs of the current file.
        # Słownik na logi dla bieżącego pliku.
        current_log_data = {
            "timestamp": timestamp,
            "image_file": image_file,
            "status": "processing",
            "start_time_utc": datetime.datetime.utcnow().isoformat()
        }

        print(f"\n--- Processing file: {image_file} ---")
        print(f"\n--- Przetwarzanie pliku: {image_file} ---")
        file_start_time = time.time()

        analysis_result_text = None
        try:
            # Pass the prompt to the image analysis function.
            # Przekazuje prompt do funkcji analizującej obraz.
            analysis_result_text = analyze_image_function(image_path, default_image_prompt, current_log_data)

            if analysis_result_text:
                print(f"SUCCESS: Analysis of file '{image_file}' completed.")
                print(f"SUKCES: Analiza pliku '{image_file}' zakończona.")

                # Save to TXT
                # Zapisz do TXT
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(analysis_result_text)
                print(f"Analysis result saved to TXT: {output_txt_path}")
                print(f"Wynik analizy zapisano do TXT: {output_txt_path}")

                # Save to PDF
                # Zapisz do PDF
                save_text_to_pdf(analysis_result_text, output_pdf_path, DEFAULT_FONT, FONT_SIZE)

                current_log_data["status"] = "SUCCESS"
            else:
                print(f"WARNING: No analysis result for file '{image_file}'.")
                print(f"OSTRZEŻENIE: Brak wyniku analizy dla pliku '{image_file}'.")
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write("[NO ANALYSIS RESULT FROM GEMINI]")
                    f.write("[BRAK WYNIKU ANALIZY Z GEMINI]")
                # Optionally create an empty or warning PDF
                # Opcjonalnie stwórz pusty lub ostrzegawczy PDF
                save_text_to_pdf("[NO ANALYSIS RESULT FROM GEMINI]\n[BRAK WYNIKU ANALIZY Z GEMINI]", output_pdf_path,
                                 DEFAULT_FONT, FONT_SIZE)
                current_log_data["status"] = "WARNING_NO_RESULT"
        except Exception as e:
            print(f"ERROR: An error occurred while processing '{image_file}': {e}")
            print(f"BŁĄD: Wystąpił błąd podczas przetwarzania '{image_file}': {e}")
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"[PROCESSING ERROR: {e}]")
                f.write(f"[BŁĄD PRZETWARZANIA: {e}]")
            # Optionally create an error PDF
            # Opcjonalnie stwórz PDF z błędem
            save_text_to_pdf(f"[PROCESSING ERROR: {e}]\n[BŁĄD PRZETWARZANIA: {e}]", output_pdf_path, DEFAULT_FONT,
                             FONT_SIZE)
            current_log_data["status"] = "ERROR"
            current_log_data["exception_details"] = str(e)

        file_end_time = time.time()
        duration = file_end_time - file_start_time
        print(f"Processing time for '{image_file}': {duration:.2f} seconds.")
        print(f"Czas przetwarzania dla '{image_file}': {duration:.2f} sekundy.")

        current_log_data["duration_seconds"] = round(duration, 2)
        current_log_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()

        # Save logs for the current file to a JSON file.
        # Zapisuje logi dla bieżącego pliku do pliku JSON.
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            json.dump(current_log_data, log_f, indent=4, ensure_ascii=False)
        print(f"Logs for file '{image_file}' saved to: {log_file_path}")
        print(f"Logi dla pliku '{image_file}' zapisano do: {log_file_path}")

        overall_logs.append(current_log_data)

        total_files_processed += 1
        total_processing_time += duration
        # Check if total_tokens is an integer before adding.
        # Sprawdza, czy total_tokens jest liczbą całkowitą przed dodaniem.
        if "total_tokens" in current_log_data and isinstance(current_log_data["total_tokens"], int):
            total_tokens_used += current_log_data["total_tokens"]

    print(f"\n--- Finished processing all image files. ---")
    print(f"\n--- Zakończono przetwarzanie wszystkich plików graficznych. ---")
    print(f"Total files processed: {total_files_processed}")
    print(f"Łącznie przetworzono plików: {total_files_processed}")
    print(f"Total execution time: {total_processing_time:.2f} seconds.")
    print(f"Całkowity czas działania: {total_processing_time:.2f} sekundy.")
    print(f"Total token usage for all files: {total_tokens_used} tokens.")
    print(f"Całkowite zużycie tokenów dla wszystkich plików: {total_tokens_used} tokenów.")

    # Optionally: Save a summary log for the entire session.
    # Opcjonalnie: Zapisz sumaryczny log dla całej sesji.
    summary_log_path = os.path.join(IMAGE_LOG_FOLDER,
                                    f"summary_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary_data = {
        "overall_summary": {
            "total_files_processed": total_files_processed,
            "total_processing_time_seconds": round(total_processing_time, 2),
            "total_tokens_used": total_tokens_used,
            "session_start_time_utc": overall_logs[0]["start_time_utc"] if overall_logs else "N/A",
            "session_end_time_utc": overall_logs[-1]["end_time_utc"] if overall_logs else "N/A"
        },
        "file_details": overall_logs
    }
    with open(summary_log_path, "w", encoding="utf-8") as sum_f:
        json.dump(summary_data, sum_f, indent=4, ensure_ascii=False)
    print(f"Session summary logs saved to: {summary_log_path}")
    print(f"Sumaryczne logi sesji zapisano do: {summary_log_path}")


if __name__ == "__main__":
    # Call the main function to process the image folder.
    # Wywołuje główną funkcję do przetwarzania folderu z obrazami.
    process_image_folder()
    print("\n--- Script image-text.py execution finished. ---")
    print("\n--- Zakończono działanie skryptu image-text.py ---")