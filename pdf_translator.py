import os
import pdfplumber
import datetime
import google.generativeai as genai
import textwrap

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

# Define the path to the Ubuntu Mono font file.
# Ensure 'UbuntuMono-Regular.ttf' is in the same location or provide a full path.
# Zdefiniuj ścieżkę do pliku czcionki Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-Regular.ttf' znajduje się w tej samej lokalizacji lub podaj pełną ścieżkę.
UBUNTU_MONO_FONT_PATH = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf"

# Create folders if they do not exist
# Utwórz foldery, jeśli nie istnieją
os.makedirs(TRANSLATIONS_FOLDER, exist_ok=True)

# Register the Ubuntu Mono font with ReportLab.
# Zarejestruj czcionkę Ubuntu Mono w ReportLab.
try:
    pdfmetrics.registerFont(TTFont('UbuntuMono', UBUNTU_MONO_FONT_PATH))
    print(f"Font 'UbuntuMono' registered successfully from {UBUNTU_MONO_FONT_PATH}.")
    print(f"Czcionka 'UbuntuMono' zarejestrowana pomyślnie z {UBUNTU_MONO_FONT_PATH}.")
    DEFAULT_FONT = "UbuntuMono"
except Exception as e:
    print(
        f"ERROR: Could not register Ubuntu Mono font from {UBUNTU_MONO_FONT_PATH}. Please ensure the file exists and is accessible. Error: {e}")
    print(
        f"BŁĄD: Nie można zarejestrować czcionki Ubuntu Mono z {UBUNTU_MONO_FONT_PATH}. Upewnij się, że plik istnieje i jest dostępny. Błąd: {e}")
    DEFAULT_FONT = "Helvetica"  # Fallback to a default font / Fallback do domyślnej czcionki

FONT_SIZE = 12  # Font size for PDF output / Rozmiar czcionki dla wyjścia PDF

# --- Google Gemini API Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual API key!
# WAŻNE: Zastąp "YOUR_API_KEY_HERE" swoim prawdziwym kluczem API!
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

generation_config = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 30,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)


# --- Function to extract ALL text from PDF / Funkcja do ekstrakcji CAŁEGO tekstu z PDF ---
def extract_full_text_from_pdf(pdf_path):
    """
    Extracts all text from all pages of a PDF file.
    Ekstrahuje cały tekst ze wszystkich stron pliku PDF.

    Args:
        pdf_path (str): Path to the PDF file. / Ścieżka do pliku PDF.

    Returns:
        str: Extracted text. Returns None on error.
             Wyekstrahowany tekst. Zwraca None w przypadku błędu.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"ERROR: Could not extract text from {pdf_path}: {e}")
        print(f"BŁĄD: Nie można wyodrębnić tekstu z {pdf_path}: {e}")
        return None
    return text.strip()


# Function to save text to PDF with titles (adapted for translations)
# Funkcja do zapisywania tekstu do PDF z tytułami (przystosowana do tłumaczeń)
def save_text_to_pdf_with_titles(text_content, output_pdf_path, font_name, font_size):
    """
    Saves the given text content to a PDF file, handling text wrapping and page breaks.
    Zapisuje podany tekst do pliku PDF, z uwzględnieniem zawijania tekstu i podziału na strony.

    Args:
        text_content (str): Text content to save. / Tekst do zapisania.
        output_pdf_path (str): Path where the PDF file should be saved. / Ścieżka, gdzie ma zostać zapisany plik PDF.
        font_name (str): Name of the registered font. / Nazwa zarejestrowanej czcionki.
        font_size (int): Font size. / Rozmiar czcionki.
    """
    print(f"Saving text to PDF: {output_pdf_path} with font '{font_name}' {font_size}pt...")
    print(f"Zapisywanie tekstu do PDF: {output_pdf_path} czcionką '{font_name}' {font_size}pkt...")
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        c.setFont(font_name, font_size)
        width, height = A4

        left_margin = 50
        top_margin = height - 50
        line_height = font_size * 1.4

        y_position = top_margin

        lines = []
        for paragraph in text_content.split('\n'):
            wrapped_lines = []
            if paragraph.strip():
                current_line = ""
                words = paragraph.split(' ')
                for word in words:
                    if pdfmetrics.stringWidth(current_line + word + ' ', font_name, font_size) < (
                            width - 2 * left_margin):
                        current_line += word + ' '
                    else:
                        wrapped_lines.append(current_line.strip())
                        current_line = word + ' '
                if current_line.strip():
                    wrapped_lines.append(current_line.strip())
            else:
                wrapped_lines.append("")
            lines.extend(wrapped_lines)

        for line in lines:
            if y_position < 50:
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


# --- Function to translate text using Gemini API / Funkcja do tłumaczenia tekstu za pomocą Gemini API ---
def translate_text_with_gemini(text_to_translate, target_language):
    """
    Translates the given text content to the target language using the Gemini 1.5 Flash model.
    Tłumaczy podany tekst na język docelowy za pomocą modelu Gemini 1.5 Flash.

    Args:
        text_to_translate (str): Text to be translated. / Tekst do przetłumaczenia.
        target_language (str): Target language ("Polski" or "Angielski"). / Język docelowy ("Polski" lub "Angielski").

    Returns:
        str: Translated text. Returns an empty string on error or if no text is provided.
             Przetłumaczony tekst. Zwraca pusty string w przypadku błędu lub braku tekstu.
    """
    print(f"Translating text to language: {target_language}...")
    print(f"Tłumaczenie tekstu na język: {target_language}...")
    if not text_to_translate.strip():
        print(f"INFO: No text to translate to {target_language}. Returning empty string.")
        print(f"INFO: Brak tekstu do tłumaczenia na {target_language}. Zwracam pusty string.")
        return ""

    # We use textwrap to split long text into chunks to avoid exceeding token limits.
    # Używamy textwrap do podziału długiego tekstu na chunki, aby uniknąć przekroczenia limitu tokenów.
    CHUNK_SIZE_TRANSLATION = 50_000
    chunks = textwrap.wrap(text_to_translate, CHUNK_SIZE_TRANSLATION, break_long_words=False, replace_whitespace=False)

    if not chunks:
        print(
            f"WARNING: Text was not split into chunks correctly for translation to {target_language}. Returning empty string.")
        print(
            f"OSTRZEŻENIE: Tekst nie został podzielony na chunki prawidłowo do tłumaczenia na {target_language}. Zwracam pusty string.")
        return ""

    translated_parts = []

    # Define the prompt prefix based on the target language for the model
    # Podziel prompt na język polski i angielski, aby model wiedział, w jakim języku ma tłumaczyć
    if target_language == "Polski":
        prompt_prefix = "Przetłumacz poniższy tekst na język polski. Zachowaj formatowanie i strukturę tekstu, w tym nagłówki i separatory z '--- POCZĄTEK PLIKU:' i '--- KONIEC PLIKU:'. Nie dodawaj żadnych dodatkowych komentarzy ani wstępów, tylko przetłumaczony tekst."
    elif target_language == "Angielski":
        prompt_prefix = "Translate the following text into English. Preserve the formatting and structure of the text, including headers and separators like '--- POCZĄTEK PLIKU:' and '--- KONIEC PLIKU:'. Do not add any extra comments or introductions, only the translated text."
    else:
        print(f"ERROR: Unsupported target language: {target_language}.")
        print(f"BŁĄD: Nieobsługiwany język docelowy: {target_language}.")
        return ""

    for i, chunk in enumerate(chunks):
        print(f"  Translating chunk {i + 1}/{len(chunks)} to {target_language}...")
        print(f"  Tłumaczenie części {i + 1}/{len(chunks)} na {target_language}...")
        prompt = f"{prompt_prefix}\n\nTekst do tłumaczenia:\n\n{chunk}"

        try:
            response = model.generate_content(prompt)

            if response.parts:
                translated_chunk_part = "".join([part.text for part in response.parts]).strip()
                translated_parts.append(translated_chunk_part)
            else:
                print(f"WARNING: Gemini returned no text for translation to {target_language} (chunk {i + 1}).")
                print(
                    f"OSTRZEŻENIE: Gemini nie zwróciło żadnego tekstu dla tłumaczenia na {target_language} (część {i + 1}).")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"  Gemini Feedback: {response.prompt_feedback}")
                    print(f"  Feedback od Gemini: {response.prompt_feedback}")
                translated_parts.append(
                    f"[TRANSLATION ERROR FOR THIS CHUNK TO {target_language}: API OR CONTENT ISSUE]\n")
                translated_parts.append(
                    f"[BŁĄD TŁUMACZENIA DLA TEJ CZĘŚCI NA {target_language}: PROBLEM Z API LUB TREŚCIĄ]\n")

        except genai.types.BlockedPromptException as e:
            print(
                f"ERROR: Translation request to {target_language} blocked by Gemini (safety settings) for chunk {i + 1}: {e}")
            print(
                f"BŁĄD: Żądanie tłumaczenia na {target_language} zablokowane przez Gemini (safety settings) dla części {i + 1}: {e}")
            translated_parts.append(f"[TRANSLATION BLOCKED BY API FOR {target_language}]\n")
            translated_parts.append(f"[TŁUMACZENIE ZABLOKOWANE PRZEZ API DLA {target_language}]\n")
        except Exception as e:
            print(
                f"CRITICAL ERROR: During communication with Google Gemini for translation to {target_language} (chunk {i + 1}): {e}")
            print(
                f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini dla tłumaczenia na {target_language} (część {i + 1}): {e}")
            translated_parts.append(f"[TRANSLATION ERROR TO {target_language} FOR THIS CHUNK: {e}]\n")
            translated_parts.append(f"[BŁĄD TŁUMACZENIA NA {target_language} DLA TEJ CZĘŚCI: {e}]\n")

    # Join the translated parts. Replace Polish separators with their English counterparts if the target language is English.
    # Połącz przetłumaczone części. Zamień polskie separatory na ich angielskie odpowiedniki, jeśli język docelowy to angielski.
    final_translated_text = "\n\n---\n\n".join(translated_parts)
    if target_language == "Angielski":
        final_translated_text = final_translated_text.replace("--- POCZĄTEK PLIKU:", "--- BEGINNING OF FILE:")
        final_translated_text = final_translated_text.replace("--- KONIEC PLIKU:", "--- END OF FILE:")
        # Replace error messages to be consistent with target language
        # Zastąp komunikaty o błędach, aby były zgodne z językiem docelowym
        final_translated_text = final_translated_text.replace(
            "[BRAK ANALIZY DLA TEJ CZĘŚCI: PROBLEM Z API LUB TREŚCIĄ]",
            "[NO ANALYSIS FOR THIS PART: API OR CONTENT ISSUE]")
        final_translated_text = final_translated_text.replace("[ANALIZA ZABLOKOWANA PRZEZ API:",
                                                              "[ANALYSIS BLOCKED BY API:")
        final_translated_text = final_translated_text.replace(
            "[ANALIZA ZABLOKOWANA ZE WZGLĘDÓW BEZPIECZEŃSTWA - BLOCKED PROMPT EXCEPTION]",
            "[ANALYSIS BLOCKED DUE TO SAFETY REASONS - BLOCKED PROMPT EXCEPTION]")
        final_translated_text = final_translated_text.replace("[BŁĄD ANALIZY DLA TEJ CZĘŚCI:",
                                                              "[ERROR ANALYZING THIS PART:")
        final_translated_text = final_translated_text.replace("[TŁUMACZENIE ZABLOKOWANE PRZEZ API DLA Polski]",
                                                              "[TRANSLATION BLOCKED BY API FOR Polish]")
        final_translated_text = final_translated_text.replace(
            "[BŁĄD TŁUMACZENIA DLA TEJ CZĘŚCI NA Polski: PROBLEM Z API LUB TREŚCIĄ]",
            "[TRANSLATION ERROR FOR THIS PART TO Polish: API OR CONTENT ISSUE]")
        final_translated_text = final_translated_text.replace("[TŁUMACZENIE ZABLOKOWANE PRZEZ API DLA Angielski]",
                                                              "[TRANSLATION BLOCKED BY API FOR English]")
        final_translated_text = final_translated_text.replace(
            "[BŁĄD TŁUMACZENIA DLA TEJ CZĘŚCI NA Angielski: PROBLEM Z API LUB TREŚCIĄ]",
            "[TRANSLATION ERROR FOR THIS PART TO English: API OR CONTENT ISSUE]")
        final_translated_text = final_translated_text.replace(
            "[BRAK GLOBALNEGO PODSUMOWANIA - MOŻLIWY PROBLEM Z API LUB BRAK TREŚCI]",
            "[NO GLOBAL SUMMARY - POSSIBLE API ISSUE OR LACK OF CONTENT]")
        final_translated_text = final_translated_text.replace("[GLOBALNE PODSUMOWANIE ZABLOKOWANE PRZEZ API]",
                                                              "[GLOBAL SUMMARY BLOCKED BY API]")
        final_translated_text = final_translated_text.replace("[GLOBALNE PODSUMOWANIE: BŁĄD ANALIZY DLA TEJ CZĘŚCI:",
                                                              "[GLOBAL SUMMARY: ERROR ANALYZING THIS PART:")
    elif target_language == "Polski":
        # For Polish translation, separators should already be in the correct form from the prompt.
        # W przypadku tłumaczenia na polski, separatory powinny być już w poprawnej formie z promptu.
        pass

    return final_translated_text


def translate_pdfs_in_folder():
    """
    Main function to translate all PDF files found in the input folder
    and save their translated versions to the translations folder.
    Główna funkcja do tłumaczenia wszystkich plików PDF znalezionych w folderze wejściowym
    i zapisywania ich przetłumaczonych wersji do folderu tłumaczeń.
    """
    print(f"Starting translation of PDF files from folder: {INPUT_FOR_TRANSLATION_FOLDER}")
    print(f"Rozpoczynam tłumaczenie plików PDF z folderu: {INPUT_FOR_TRANSLATION_FOLDER}")

    pdf_files = sorted([f for f in os.listdir(INPUT_FOR_TRANSLATION_FOLDER) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print("INFO: No PDF files found in the input folder for translation.")
        print("INFO: Brak plików PDF w folderze wejściowym do tłumaczenia.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_FOR_TRANSLATION_FOLDER, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n--- Processing file for translation: {pdf_file} ---")
        print(f"\n--- Przetwarzanie pliku do tłumaczenia: {pdf_file} ---")

        extracted_text = extract_full_text_from_pdf(pdf_path)

        if extracted_text is None:
            print(f"ERROR: Skipping file '{pdf_file}' due to text extraction issues.")
            print(f"BŁĄD: Pomijam plik '{pdf_file}' z powodu problemów z ekstrakcją tekstu.")
            continue

        if not extracted_text.strip():
            print(f"INFO: File '{pdf_file}' is empty or contains no text. Skipping translation.")
            print(f"INFO: Plik '{pdf_file}' jest pusty lub nie zawiera tekstu. Pomijam tłumaczenie.")
            continue

        # Translate to English
        # Tłumaczenie na angielski
        translated_text_en = translate_text_with_gemini(extracted_text, "Angielski")
        if translated_text_en:
            output_translated_en_pdf_path = os.path.join(TRANSLATIONS_FOLDER, f"{base_name}_{timestamp}_EN.pdf")
            save_text_to_pdf_with_titles(translated_text_en, output_translated_en_pdf_path, DEFAULT_FONT, FONT_SIZE)
            print(f"Successfully saved translated document to English at: {output_translated_en_pdf_path}")
            print(f"Pomyślnie zapisano przetłumaczony dokument na angielski do: {output_translated_en_pdf_path}")
        else:
            print(f"WARNING: Failed to translate document '{pdf_file}' to English.")
            print(f"OSTRZEŻENIE: Nie udało się przetłumaczyć dokumentu '{pdf_file}' na angielski.")

        # Translate to Polish
        # Tłumaczenie na polski
        translated_text_pl = translate_text_with_gemini(extracted_text, "Polski")
        if translated_text_pl:
            output_translated_pl_pdf_path = os.path.join(TRANSLATIONS_FOLDER, f"{base_name}_{timestamp}_PL.pdf")
            save_text_to_pdf_with_titles(translated_text_pl, output_translated_pl_pdf_path, DEFAULT_FONT, FONT_SIZE)
            print(f"Successfully saved translated document to Polish at: {output_translated_pl_pdf_path}")
            print(f"Pomyślnie zapisano przetłumaczony dokument na polski do: {output_translated_pl_pdf_path}")
        else:
            print(f"WARNING: Failed to translate document '{pdf_file}' to Polish.")
            print(f"OSTRZEŻENIE: Nie udało się przetłumaczyć dokumentu '{pdf_file}' na polski.")

        print(f"--- Finished translating file: {pdf_file} ---")
        print(f"--- Zakończono tłumaczenie pliku: {pdf_file} ---")

    print("\n--- Finished the translation process for all PDF files. ---")
    print("\n--- Zakończono proces tłumaczenia wszystkich plików PDF. ---")


if __name__ == "__main__":
    # Entry point of the program. Calls the PDF translation function.
    # Punkt wejścia programu. Wywołuje funkcję tłumaczenia PDF-ów.
    translate_pdfs_in_folder()
