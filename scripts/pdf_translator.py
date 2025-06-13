import os
import pdfplumber
import datetime
import google.generativeai as genai
import textwrap
import json

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Path Configuration / Konfiguracja ścieżek ---
# Source folder for PDF files to be translated (e.g., FOR_ANALYSIS folder)
# Folder źródłowy dla plików PDF do tłumaczenia (np. folder FOR_ANALYSIS)
INPUT_FOR_TRANSLATION_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/FOR_ANALYSIS"
# Destination folder for translated PDF files
# Folder docelowy dla przetłumaczonych plików PDF
TRANSLATIONS_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/TRANSLATIONS"
# Log folder for usage summaries
# Folder na logi dla podsumowań zużycia
LOG_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/LOGS"

# Define the path to the Ubuntu Mono font file.
# Ensure 'UbuntuMono-Regular.ttf' and 'UbuntuMono-Bold.ttf' are in the same location or provide full paths.
# Zdefiniuj ścieżkę do plików czcionek Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-Regular.ttf' i 'UbuntuMono-Bold.ttf' znajdują się w tej samej lokalizacji lub podaj pełne ścieżki.
UBUNTU_MONO_REGULAR_FONT_PATH = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf"
UBUNTU_MONO_BOLD_FONT_PATH = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Bold.ttf"

# Create folders if they do not exist
# Utwórz foldery, jeśli nie istnieją
os.makedirs(TRANSLATIONS_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Register the Ubuntu Mono font with ReportLab.
# Zarejestruj czcionki Ubuntu Mono w ReportLab.
FONT_NAME_REGULAR = "UbuntuMono-Regular"
FONT_NAME_BOLD = "UbuntuMono-Bold"

try:
    pdfmetrics.registerFont(TTFont(FONT_NAME_REGULAR, UBUNTU_MONO_REGULAR_FONT_PATH))
    print(f"Font '{FONT_NAME_REGULAR}' registered successfully from {UBUNTU_MONO_REGULAR_FONT_PATH}.")
    print(f"Czcionka '{FONT_NAME_REGULAR}' zarejestrowana pomyślnie z {UBUNTU_MONO_REGULAR_FONT_PATH}.")

    pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, UBUNTU_MONO_BOLD_FONT_PATH))
    print(f"Font '{FONT_NAME_BOLD}' registered successfully from {UBUNTU_MONO_BOLD_FONT_PATH}.")
    print(f"Czcionka '{FONT_NAME_BOLD}' zarejestrowana pomyślnie z {UBUNTU_MONO_BOLD_FONT_PATH}.")

except Exception as e:
    print(
        f"ERROR: Could not register Ubuntu Mono fonts. Please ensure the files exist and are accessible. Error: {e}")
    print(
        f"BŁĄD: Nie można zarejestrować czcionek Ubuntu Mono. Upewnij się, że pliki istnieją i są dostępne. Błąd: {e}")
    FONT_NAME_REGULAR = "Helvetica"  # Fallback to a default font
    FONT_NAME_BOLD = "Helvetica-Bold"  # Fallback to a default bold font
    # Attempt to register Helvetica-Bold if not already
    # Spróbuj zarejestrować Helvetica-Bold, jeśli jeszcze nie jest zarejestrowana
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, FONT_NAME_BOLD))
    except Exception:
        pass  # Ignore if default bold is also not there or already registered
        # Ignoruj, jeśli domyślna pogrubiona czcionka również nie jest dostępna lub jest już zarejestrowana

    # Exit the program if the font is critical and not found
    # Wyjście z programu, jeśli czcionka jest krytyczna i nie została znaleziona
    exit("Exiting: Critical font(s) not found or accessible. Please check font paths.")

FONT_SIZE = 12  # Font size for PDF output / Rozmiar czcionki dla wyjścia PDF
LINE_HEIGHT = FONT_SIZE * 1.2  # Adjusted line height for better readability / Dostosowana wysokość linii dla lepszej czytelności
default_margin = 50  # Default margin for PDF content / Domyślny margines dla zawartości PDF

# --- Google Gemini API Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual API key!
# WAŻNE: Zastąp "YOUR_API_KEY_HERE" swoim prawdziwym kluczem API!
GOOGLE_API_KEY = "*****"  # Your asterisks are back! / Twoje gwiazdki są z powrotem!
genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL_NAME = "gemini-1.5-flash"  # Changed to GEMINI_MODEL_NAME for consistency / Zmienione na GEMINI_MODEL_NAME dla spójności

generation_config = {
    "temperature": 0.1,  # Lower temperature for more focused output / Niższa temperatura dla bardziej skupionego wyniku
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    # Increased token limit for more flexibility / Zwiększono limit tokenów dla większej elastyczności
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,  # Changed to GEMINI_MODEL_NAME
    generation_config=generation_config,
    safety_settings=safety_settings
)

CHUNK_SIZE_TRANSLATION = 50_000  # Still relevant if a single page is very long / Nadal istotne, jeśli pojedyncza strona jest bardzo długa


# --- Helper Functions / Funkcje Pomocnicze ---

def draw_header(c, page_width, current_time_str, model_name, prompt_text, font_name_regular, font_name_bold, font_size,
                language_code):
    """
    Draws the header content on the PDF canvas with bold titles, localized by language_code.
    Rysuje zawartość nagłówka na płótnie PDF z pogrubionymi tytułami, zlokalizowanymi według kodu języka.
    """
    header_x_start = 50
    header_y_start = A4[1] - 30
    header_line_height = font_size * 1.2

    # Localized header titles
    # Zlokalizowane tytuły nagłówków
    if language_code == "PL":
        translated_with_ai_label = "Przetłumaczono z AI"
        ai_model_version_label = "Wersja modelu AI"
        time_stamp_label = "Znacznik czasu"
        prompt_label = "Zapytanie (Prompt)"
        translation_label = "Tłumaczenie"
        translation_continued_label = "Tłumaczenie (ciąg dalszy)"  # For subsequent pages
    else:  # Default to English / Domyślnie na angielski
        translated_with_ai_label = "Translated with AI"
        ai_model_version_label = "AI Model Version"
        time_stamp_label = "Time Stamp"
        prompt_label = "Prompt"
        translation_label = "Translation"
        translation_continued_label = "Translation (continued)"

    # Title "Translated with AI"
    # Tytuł "Przetłumaczono z AI"
    c.setFont(font_name_bold, font_size)
    c.drawString(header_x_start, header_y_start, translated_with_ai_label)

    # AI Model Version
    # Wersja modelu AI
    c.setFont(font_name_bold, font_size)
    c.drawString(header_x_start, header_y_start - header_line_height, f"{ai_model_version_label}:")
    c.setFont(font_name_regular, font_size)
    c.drawString(header_x_start + pdfmetrics.stringWidth(f"{ai_model_version_label}: ", font_name_bold, font_size),
                 header_y_start - header_line_height, model_name)

    # Time Stamp
    # Znacznik czasu
    c.setFont(font_name_bold, font_size)
    c.drawString(header_x_start, header_y_start - 2 * header_line_height, f"{time_stamp_label}:")
    c.setFont(font_name_regular, font_size)
    c.drawString(header_x_start + pdfmetrics.stringWidth(f"{time_stamp_label}: ", font_name_bold, font_size),
                 header_y_start - 2 * header_line_height, current_time_str)

    # Prompt
    # Zapytanie (Prompt)
    c.setFont(font_name_bold, font_size)
    c.drawString(header_x_start, header_y_start - 3 * header_line_height, f"{prompt_label}:")

    # Wrap and draw the prompt text (regular font)
    # Zawijanie i rysowanie tekstu prompta (zwykła czcionka)
    c.setFont(font_name_regular, font_size)
    prompt_width = page_width - 2 * header_x_start
    wrapped_prompt = textwrap.wrap(prompt_text, width=int(prompt_width / (font_size * 0.6)))

    for i, line in enumerate(wrapped_prompt):
        c.drawString(header_x_start, header_y_start - (4 + i) * header_line_height, line)

    # Adjust y_position after header and return localized labels
    # Dostosuj pozycję y po nagłówku i zwróć zlokalizowane etykiety
    return header_y_start - (4 + len(
        wrapped_prompt)) * header_line_height - LINE_HEIGHT, translation_label, translation_continued_label


# Function to save a list of translated pages to PDF / Funkcja do zapisywania listy przetłumaczonych stron do PDF
def save_translated_pages_to_pdf(list_of_page_texts, output_pdf_path, font_name_regular, font_name_bold, font_size,
                                 target_language):
    """
    Saves a list of text contents (each representing a page) to a PDF file,
    with a header on the first page only and localized headings.
    Zapisuje listę zawartości tekstowych (każda reprezentująca stronę) do pliku PDF,
    z nagłówkiem tylko na pierwszej stronie i zlokalizowanymi nagłówkami.
    """
    print(f"Saving translated pages to PDF: {output_pdf_path} with font '{font_name_regular}' {font_size}pt...")
    print(
        f"Zapisywanie przetłumaczonych stron do PDF: {output_pdf_path} czcionką '{font_name_regular}' {font_size}pkt...")
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        width, height = A4

        left_margin = 50

        current_time_str_for_header = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Define the prompt text for the header based on the target language
        # Zdefiniuj tekst prompta dla nagłówka na podstawie języka docelowego
        if target_language == "Polski":
            prompt_text_for_header = "Przetłumacz poniższy tekst na język polski. Zachowaj formatowanie i strukturę tekstu, w tym podziały wierszy. Nie dodawaj żadnych dodatkowych komentarzy ani wstępów, tylko przetłumaczony tekst."
            language_code_for_header = "PL"
        elif target_language == "Angielski":
            prompt_text_for_header = "Translate the following text into English. Preserve the formatting and structure of the text, including line breaks. Do not add any extra comments or introductions, only the translated text."
            language_code_for_header = "EN"
        else:
            prompt_text_for_header = "Translate the following text. Preserve formatting and structure. No extra comments."
            language_code_for_header = "EN"  # Default to English for unknown / Domyślnie na angielski dla nieznanych

        translation_label_text = ""
        translation_continued_label_text = ""

        for page_num, page_text in enumerate(list_of_page_texts):
            if page_num > 0:  # Start new PDF page for subsequent pages
                # Rozpocznij nową stronę PDF dla kolejnych stron
                c.showPage()
                c.setFont(font_name_regular, font_size)
                y_position = height - default_margin  # Reset y_position for new page / Zresetuj pozycję y dla nowej strony
                # Add "Translation (continued)" if defined and not first page
                # Dodaj "Tłumaczenie (ciąg dalszy)", jeśli zdefiniowano i nie jest to pierwsza strona
                if translation_continued_label_text:
                    c.setFont(font_name_bold, font_size)  # Make "Translation (continued)" bold
                    # Ustaw "Tłumaczenie (ciąg dalszy)" na pogrubione
                    c.drawString(left_margin, y_position, translation_continued_label_text)
                    y_position -= LINE_HEIGHT * 2
                    c.setFont(font_name_regular, font_size)  # Switch back to regular for content
                    # Przełącz z powrotem na zwykłą czcionkę dla treści
            else:  # First page, draw header
                # Pierwsza strona, narysuj nagłówek
                y_position, translation_label_text, translation_continued_label_text = draw_header(c, width,
                                                                                                   current_time_str_for_header,
                                                                                                   GEMINI_MODEL_NAME,
                                                                                                   prompt_text_for_header,
                                                                                                   font_name_regular,
                                                                                                   font_name_bold,
                                                                                                   font_size,
                                                                                                   language_code_for_header)
                c.setFont(font_name_regular, font_size)  # Set font for content after header
                # Ustaw czcionkę dla treści po nagłówku

                # Add "Translation" heading for the actual content section (bold, only on first page)
                # Dodaj nagłówek "Tłumaczenie" dla rzeczywistej sekcji treści (pogrubiony, tylko na pierwszej stronie)
                c.setFont(font_name_bold, font_size)
                c.drawString(left_margin, y_position, translation_label_text)
                y_position -= LINE_HEIGHT * 2
                c.setFont(font_name_regular, font_size)

            lines = []
            for paragraph in page_text.split('\n'):
                wrapped_lines = []
                if paragraph.strip():
                    current_line = ""
                    words = paragraph.split(' ')
                    for word in words:
                        # Check if adding the next word exceeds page width
                        # Sprawdź, czy dodanie następnego słowa przekracza szerokość strony
                        if pdfmetrics.stringWidth(current_line + word + ' ', font_name_regular, font_size) < (
                                width - 2 * left_margin):
                            current_line += word + ' '
                        else:
                            wrapped_lines.append(current_line.strip())
                            current_line = word + ' '
                    if current_line.strip():
                        wrapped_lines.append(current_line.strip())
                else:
                    wrapped_lines.append("")  # Preserve empty lines/paragraphs / Zachowaj puste linie/akapity
                lines.extend(wrapped_lines)

            for line in lines:
                if y_position < 50:  # If text overflows current PDF page, create new PDF page
                    # Jeśli tekst przepełnia bieżącą stronę PDF, utwórz nową stronę PDF
                    c.showPage()
                    c.setFont(font_name_regular, font_size)
                    y_position = height - default_margin
                    # Add "Translation (continued)" if defined
                    # Dodaj "Tłumaczenie (ciąg dalszy)", jeśli zdefiniowano
                    if translation_continued_label_text:
                        c.setFont(font_name_bold, font_size)
                        c.drawString(left_margin, y_position, translation_continued_label_text)
                        y_position -= LINE_HEIGHT * 2
                        c.setFont(font_name_regular, font_size)

                c.drawString(left_margin, y_position, line)
                y_position -= LINE_HEIGHT

        c.save()
        print(f"Successfully saved PDF to: {output_pdf_path}")
        print(f"Pomyślnie zapisano PDF do: {output_pdf_path}")
    except Exception as e:
        print(f"ERROR: Could not save text to PDF '{output_pdf_path}': {e}")
        print(f"BŁĄD: Nie można zapisać tekstu do PDF '{output_pdf_path}': {e}")


# Function to extract text page by page from PDF / Funkcja do ekstrakcji tekstu strona po stronie z PDF ---
def extract_text_page_by_page(pdf_path):
    """
    Extracts text from each page of a PDF file, returning a list of page texts.
    Ekstrahuje tekst z każdej strony pliku PDF, zwracając listę tekstów stron.
    """
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())
                else:
                    pages_text.append("")  # Append empty string for empty pages to preserve page count
                    # Dołącz pusty ciąg dla pustych stron, aby zachować liczbę stron
                    print(f"  INFO: Page {i + 1} is empty or contains no text.")
                    print(f"  INFO: Strona {i + 1} jest pusta lub nie zawiera tekstu.")
    except Exception as e:
        print(f"ERROR: Could not extract text from {pdf_path}: {e}")
        print(f"BŁĄD: Nie można wyodrębnić tekstu z {pdf_path}: {e}")
        return None
    return pages_text


# --- Function to translate text using Gemini API / Funkcja do tłumaczenia tekstu za pomocą Gemini API ---
def translate_text_with_gemini(list_of_texts_to_translate, target_language, total_input_tokens_ref,
                               total_output_tokens_ref):
    """
    Translates a list of text chunks (pages) to the target language using the Gemini 1.5 Flash model.
    Updates total_input_tokens_ref and total_output_tokens_ref with token counts.
    Tłumaczy listę fragmentów tekstu (stron) na język docelowy za pomocą modelu Gemini 1.5 Flash.
    Aktualizuje total_input_tokens_ref i total_output_tokens_ref o liczbę tokenów.
    """
    print(f"Translating {len(list_of_texts_to_translate)} text chunks to language: {target_language}...")
    print(f"Tłumaczenie {len(list_of_texts_to_translate)} fragmentów tekstu na język: {target_language}...")
    if not list_of_texts_to_translate:
        print(f"INFO: No text chunks to translate to {target_language}. Returning empty list.")
        print(f"INFO: Brak fragmentów tekstu do tłumaczenia na {target_language}. Zwracam pustą listę.")
        return []

    translated_pages = []  # To store translated text for each page / Do przechowywania przetłumaczonego tekstu dla każdej strony

    # Define the prompt prefix based on the target language for the model
    # Podziel prompt na język polski i angielski, aby model wiedział, w jakim języku ma tłumaczyć
    if target_language == "Polski":
        prompt_prefix = "Przetłumacz poniższy tekst na język polski. Zachowaj formatowanie i strukturę tekstu, w tym podziały wierszy. Nie dodawaj żadnych dodatkowych komentarzy ani wstępów, tylko przetłumaczony tekst."
    elif target_language == "Angielski":
        prompt_prefix = "Translate the following text into English. Preserve the formatting and structure of the text, including line breaks. Do not add any extra comments or introductions, only the translated text."
    else:
        print(f"ERROR: Unsupported target language: {target_language}.")
        print(f"BŁĄD: Nieobsługiwany język docelowy: {target_language}.")
        return []

    for i, page_text in enumerate(list_of_texts_to_translate):
        if not page_text.strip():
            translated_pages.append("")  # Keep empty pages empty / Pozostaw puste strony pustymi
            continue

        # If a single page is too long, chunk it
        # Jeśli pojedyncza strona jest zbyt długa, podziel ją na fragmenty
        chunks_for_this_page = textwrap.wrap(page_text, CHUNK_SIZE_TRANSLATION, break_long_words=False,
                                             replace_whitespace=False)

        current_page_translated_parts = []
        for j, chunk in enumerate(chunks_for_this_page):
            print(f"  Translating page {i + 1} (chunk {j + 1}/{len(chunks_for_this_page)}) to {target_language}...")
            print(
                f"  Tłumaczenie strony {i + 1} (fragment {j + 1}/{len(chunks_for_this_page)}) na {target_language}...")
            prompt = f"{prompt_prefix}\n\nTekst do tłumaczenia:\n\n{chunk}"

            current_chunk_input_tokens = 0
            current_chunk_output_tokens = 0

            try:
                current_chunk_input_tokens = model.count_tokens(prompt).total_tokens
                total_input_tokens_ref[0] += current_chunk_input_tokens
                print(f"  Estimated input tokens for this chunk: {current_chunk_input_tokens}")
                print(f"  Szacowane tokeny wejściowe dla tej części: {current_chunk_input_tokens}")

                response = model.generate_content(prompt)

                if response.parts:
                    translated_chunk_part = "".join([part.text for part in response.parts]).strip()
                    current_page_translated_parts.append(translated_chunk_part)

                    try:
                        # Attempt to get usageMetadata if available
                        # Spróbuj pobrać usageMetadata, jeśli dostępne
                        if hasattr(response, '_result') and 'usageMetadata' in response._result:
                            usage_metadata = response._result['usageMetadata']
                            current_chunk_output_tokens = usage_metadata.get('candidatesTokenCount', 0)
                            print(
                                f"  Generated output tokens for this chunk (from usageMetadata): {current_chunk_output_tokens}")
                            print(
                                f"  Wygenerowane tokeny wyjściowe dla tej części (z usageMetadata): {current_chunk_output_tokens}")
                        else:
                            # Fallback if usageMetadata is not present
                            # Wróć do alternatywy, jeśli usageMetadata nie jest obecne
                            print(
                                "  No usageMetadata in response, despite content. Estimating output tokens from text.")
                            print(
                                "  Brak usageMetadata w odpowiedzi, mimo obecności treści. Oszacowanie tokenów wyjściowych na podstawie tekstu.")
                            current_chunk_output_tokens = model.count_tokens(translated_chunk_part).total_tokens
                            print(
                                f"  Estimated output tokens for this chunk (from text): {current_chunk_output_tokens}")
                            print(
                                f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {current_chunk_output_tokens}")
                    except Exception as usage_e:
                        print(f"ERROR getting usageMetadata for chunk {j + 1}: {usage_e}")
                        print(f"BŁĄD podczas pobierania usageMetadata dla części {j + 1}: {usage_e}")
                        print("  Estimating output tokens from generated text.")
                        print("  Oszacowanie tokenów wyjściowych na podstawie wygenerowanego tekstu.")
                        current_chunk_output_tokens = model.count_tokens(translated_chunk_part).total_tokens
                        print(f"  Estimated output tokens for this chunk (from text): {current_chunk_output_tokens}")
                        print(f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {current_chunk_output_tokens}")
                else:
                    print(
                        f"WARNING: Gemini returned no text for translation of page {i + 1} (chunk {j + 1}) to {target_language}.")
                    print(
                        f"OSTRZEŻENIE: Gemini nie zwróciło żadnego tekstu dla tłumaczenia strony {i + 1} (fragment {j + 1}) na {target_language}.")
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        print(f"  Gemini Feedback: {response.prompt_feedback}")
                        print(f"  Feedback od Gemini: {response.prompt_feedback}")
                    current_page_translated_parts.append(
                        f"[TRANSLATION ERROR FOR PAGE {i + 1} CHUNK {j + 1} TO {target_language}: API OR CONTENT ISSUE]\n")
                    current_page_translated_parts.append(
                        f"[BŁĄD TŁUMACZENIA DLA STRONY {i + 1} FRAGMENTU {j + 1} NA {target_language}: PROBLEM Z API LUB TREŚCIĄ]\n")

                total_output_tokens_ref[0] += current_chunk_output_tokens

            except genai.types.BlockedPromptException as e:
                print(
                    f"ERROR: Translation request for page {i + 1} (chunk {j + 1}) to {target_language} blocked by Gemini (safety settings): {e}")
                print(
                    f"BŁĄD: Żądanie tłumaczenia dla strony {i + 1} (fragment {j + 1}) na {target_language} zablokowane przez Gemini (safety settings): {e}")
                current_page_translated_parts.append(
                    f"[TRANSLATION BLOCKED BY API FOR PAGE {i + 1} CHUNK {j + 1} TO {target_language}]\n")
                current_page_translated_parts.append(
                    f"[TŁUMACZENIE ZABLOKOWANE PRZEZ API DLA STRONY {i + 1} FRAGMENTU {j + 1} NA {target_language}]\n")
            except Exception as e:
                print(
                    f"CRITICAL ERROR: During communication with Google Gemini for translation of page {i + 1} (chunk {j + 1}) to {target_language}: {e}")
                print(
                    f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini dla tłumaczenia strony {i + 1} (fragment {j + 1}) na {target_language}: {e}")
                current_page_translated_parts.append(f"[TRANSLATION ERROR TO {target_language} FOR THIS CHUNK: {e}]\n")
                current_page_translated_parts.append(
                    f"[BŁĄD TŁUMACZENIA DLA STRONY {i + 1} FRAGMENTU {j + 1} NA {target_language}: {e}]\n")

        translated_pages.append("\n".join(current_page_translated_parts))

    return translated_pages


def write_usage_summary(total_files, total_input_tokens, total_output_tokens, total_duration):
    """
    Writes a summary of Gemini API usage and processing time to a log file.
    Zapisuje podsumowanie zużycia API Gemini i czasu przetwarzania do pliku logu.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_FOLDER, f"gemini_translation_usage_summary_{timestamp}.txt")

    summary_content = (
        f"--- BLOX-TAK-GEMINI Translation Usage Summary ---\n"
        f"--- Podsumowanie zużycia tłumaczenia BLOX-TAK-GEMINI ---\n"
        f"Run Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Data i czas uruchomienia: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Processed PDF files: {total_files}\n"
        f"Przetworzone plików PDF: {total_files}\n"
        f"Gemini Model Used: {GEMINI_MODEL_NAME}\n"
        f"Użyty model Gemini: {GEMINI_MODEL_NAME}\n"
        f"Total Input Tokens (prompt): {total_input_tokens}\n"
        f"Całkowita liczba tokenów wejściowych (prompt): {total_input_tokens}\n"
        f"Total Output Tokens (generation): {total_output_tokens}\n"
        f"Całkowita liczba tokenów wyjściowych (generacja): {total_output_tokens}\n"
        f"Total Processing Duration: {total_duration:.2f} seconds\n"
        f"Łączny czas przetwarzania: {total_duration:.2f} sekund\n"
        f"--------------------------------------------------\n"
        f"NOTE: If output tokens are 0 or suspiciously low despite generated content, "
        f"this may indicate an API issue, content blocking by Gemini safety filters, "
        f"or an error in retrieving usageMetadata. Check console logs above for 'Gemini Feedback' or 'Safety Ratings'.\n"
        f"UWAGA: Jeśli tokeny wyjściowe wynoszą 0 lub są podejrzanie niskie mimo wygenerowanej treści, "
        f"może to oznaczać problem z API lub blokadę treści przez filtry bezpieczeństwa Gemini, "
        f"lub błąd w pobieraniu usageMetadata. Sprawdź logi konsoli powyżej dla 'Feedback od Gemini' lub 'Safety Ratings'.\n"
    )

    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
        print(f"\n--- Usage summary saved to: {log_file_path} ---")
        print(f"\n--- Podsumowanie zużycia zapisano do: {log_file_path} ---")
    except Exception as e:
        print(f"ERROR: Could not save usage summary file: {e}")
        print(f"BŁĄD: Nie można zapisać pliku podsumowania zużycia: {e}")


def translate_pdfs_in_folder():
    """
    Main function to translate all PDF files found in the input folder page-by-page
    and save their translated versions to the translations folder.
    Główna funkcja do tłumaczenia wszystkich plików PDF znalezionych w folderze wejściowym strona po stronie
    i zapisywania ich przetłumaczonych wersji do folderu tłumaczeń.
    """
    print(f"Starting page-by-page translation of PDF files from folder: {INPUT_FOR_TRANSLATION_FOLDER}")
    print(f"Rozpoczynam tłumaczenie strona po stronie plików PDF z folderu: {INPUT_FOR_TRANSLATION_FOLDER}")

    pdf_files = sorted([f for f in os.listdir(INPUT_FOR_TRANSLATION_FOLDER) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print("INFO: No PDF files found in the input folder for translation.")
        print("INFO: Brak plików PDF w folderze wejściowym do tłumaczenia.")
        return

    overall_total_input_tokens = [
        0]  # Use mutable list to pass by reference / Użyj listy zmiennej, aby przekazać przez referencję
    overall_total_output_tokens = [
        0]  # Use mutable list to pass by reference / Użyj listy zmiennej, aby przekazać przez referencję
    processed_files_count = 0

    start_time = datetime.datetime.now()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_FOR_TRANSLATION_FOLDER, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        # We use a unique timestamp for the output file names for each processed file
        # Używamy unikalnego timestampu dla nazw plików wyjściowych dla każdego przetwarzanego pliku
        unique_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        print(f"\n--- Processing file for translation: {pdf_file} ---")
        print(f"\n--- Przetwarzanie pliku do tłumaczenia: {pdf_file} ---")

        # Extract text page by page
        # Ekstrakcja tekstu strona po stronie
        extracted_pages_text = extract_text_page_by_page(pdf_path)

        if extracted_pages_text is None:
            print(f"ERROR: Skipping file '{pdf_file}' due to text extraction issues.")
            print(f"BŁĄD: Pomijam plik '{pdf_file}' z powodu problemów z ekstrakcją tekstu.")
            continue

        if not any(page_text.strip() for page_text in
                   extracted_pages_text):  # Check if any page has content / Sprawdzamy, czy jakakolwiek strona ma zawartość
            print(f"INFO: File '{pdf_file}' is empty or contains no readable text. Skipping translation.")
            print(f"INFO: Plik '{pdf_file}' jest pusty lub nie zawiera czytelnego tekstu. Pomijam tłumaczenie.")
            continue

        # Translate page by page to English
        # Tłumaczenie strona po stronie na angielski
        translated_pages_en = translate_text_with_gemini(extracted_pages_text, "Angielski", overall_total_input_tokens,
                                                         overall_total_output_tokens)
        if translated_pages_en:
            output_translated_en_pdf_path = os.path.join(TRANSLATIONS_FOLDER, f"{base_name}_{unique_timestamp}_EN.pdf")
            save_translated_pages_to_pdf(translated_pages_en, output_translated_en_pdf_path, FONT_NAME_REGULAR,
                                         FONT_NAME_BOLD, FONT_SIZE, "Angielski")
            print(f"Successfully saved translated document to English at: {output_translated_en_pdf_path}")
            print(f"Pomyślnie zapisano przetłumaczony dokument na angielski do: {output_translated_en_pdf_path}")
        else:
            print(f"WARNING: Failed to translate document '{pdf_file}' to English.")
            print(f"OSTRZEŻENIE: Nie udało się przetłumaczyć dokumentu '{pdf_file}' na angielski.")

        # Translate page by page to Polish
        # Tłumaczenie strona po stronie na polski
        translated_pages_pl = translate_text_with_gemini(extracted_pages_text, "Polski", overall_total_input_tokens,
                                                         overall_total_output_tokens)
        if translated_pages_pl:
            output_translated_pl_pdf_path = os.path.join(TRANSLATIONS_FOLDER, f"{base_name}_{unique_timestamp}_PL.pdf")
            save_translated_pages_to_pdf(translated_pages_pl, output_translated_pl_pdf_path, FONT_NAME_REGULAR,
                                         FONT_NAME_BOLD, FONT_SIZE, "Polski")
            print(f"Successfully saved translated document to Polish at: {output_translated_pl_pdf_path}")
            print(f"Pomyślnie zapisano przetłumaczony dokument na polski do: {output_translated_pl_pdf_path}")
        else:
            print(f"WARNING: Failed to translate document '{pdf_file}' to Polish.")
            print(f"OSTRZEŻENIE: Nie udało się przetłumaczyć dokumentu '{pdf_file}' na polski.")

        print(f"--- Finished translating file: {pdf_file} ---")
        print(f"--- Zakończono tłumaczenie pliku: {pdf_file} ---")
        processed_files_count += 1

    end_time = datetime.datetime.now()
    total_processing_duration = (end_time - start_time).total_seconds()

    print("\n--- Finished the translation process for all PDF files. ---")
    print("\n--- Zakończono proces tłumaczenia wszystkich plików PDF. ---")

    # Write overall usage summary
    # Zapisz ogólne podsumowanie zużycia
    write_usage_summary(processed_files_count, overall_total_input_tokens[0], overall_total_output_tokens[0],
                        total_processing_duration)


if __name__ == "__main__":
    # Entry point of the program. Calls the PDF translation function.
    # Punkt wejścia programu. Wywołuje funkcję tłumaczenia PDF-ów.
    translate_pdfs_in_folder()