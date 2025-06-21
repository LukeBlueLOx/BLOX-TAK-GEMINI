import os
import pdfplumber
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Path Configuration / Konfiguracja ścieżek ---
# Source folder for PDF files
# Folder źródłowy dla plików PDF
PDF_SOURCE_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/TEMP"
# Destination folder for the output PDF file intended for analysis
# Folder docelowy dla wynikowego pliku PDF przeznaczonego do analizy
OUTPUT_FOR_ANALYSIS_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/FOR_ANALYSIS"

# Define the path to the Ubuntu Mono font file.
# Ensure 'UbuntuMono-Regular.ttf' is in the same location or provide a full path.
# Zdefiniuj ścieżkę do pliku czcionki Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-Regular.ttf' znajduje się w tej samej lokalizacji lub podaj pełną ścieżkę.
UBUNTU_MONO_FONT_PATH = os.path.join(os.path.dirname(__file__),
                                     "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf")

# Create folders if they do not exist
# Utwórz foldery, jeśli nie istnieją
os.makedirs(PDF_SOURCE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOR_ANALYSIS_FOLDER, exist_ok=True)

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


# --- Function to extract text from SELECTED PAGES / Funkcja do ekstrakcji tekstu z WYBRANYCH STRON ---
def extract_selected_pages_from_pdf(pdf_path, start_page, end_page):
    """
    Extracts text from a specified page range of a PDF file.
    Ekstrahuje tekst z określonego zakresu stron pliku PDF.

    Args:
        pdf_path (str): Path to the PDF file. / Ścieżka do pliku PDF.
        start_page (int): Starting page number (1-based). / Numer strony początkowej (od 1).
        end_page (int): Ending page number (inclusive). / Numer strony końcowej (włącznie).

    Returns:
        str: Extracted text. Returns an empty string if no text, or None on error.
             Wyekstrahowany tekst. Zwraca pusty string, jeśli nie ma tekstu lub None w przypadku błędu.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)

            # Adjust page indices to Python's 0-based range and ensure they are within bounds
            # Dostosuj indeksy stron do zakresu Pythona (0-based) i upewnij się, że są w granicach
            actual_start_index = max(0, start_page - 1)
            actual_end_index = min(num_pages, end_page)

            if actual_start_index >= num_pages:
                print(
                    f"WARNING: Start page {start_page} exceeds the number of PDF pages ({num_pages}). Returning empty text.")
                print(
                    f"OSTRZEŻENIE: Strona początkowa {start_page} wykracza poza liczbę stron PDF ({num_pages}). Zwracam pusty tekst.")
                return ""
            if actual_start_index >= actual_end_index:
                print(f"WARNING: Page range ({start_page}-{end_page}) is invalid or empty. Returning empty text.")
                print(
                    f"OSTRZEŻENIE: Zakres stron ({start_page}-{end_page}) jest nieprawidłowy lub pusty. Zwracam pusty tekst.")
                return ""

            print(
                f"Extracting pages from {actual_start_index + 1} to {actual_end_index} from file '{os.path.basename(pdf_path)}'...")
            print(
                f"Ekstrakcja stron od {actual_start_index + 1} do {actual_end_index} z pliku '{os.path.basename(pdf_path)}'...")

            for i in range(actual_start_index, actual_end_index):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    print(f"  INFO: Page {i + 1} is empty or contains no text.")
                    print(f"  INFO: Strona {i + 1} jest pusta lub nie zawiera tekstu.")
    except Exception as e:
        print(f"ERROR: Could not extract text from {pdf_path} for pages {start_page}-{end_page}: {e}")
        print(f"BŁĄD: Nie można wyodrębnić tekstu z {pdf_path} dla stron {start_page}-{end_page}: {e}")
        return None
    return text.strip()


# --- Function for interactive page range input / Funkcja do interaktywnego pobierania zakresu stron ---
def get_page_range_input(pdf_file_name, total_pages):
    """
    Interactively prompts the user for the page range to process for a given PDF file.
    Interaktywnie prosi użytkownika o zakres stron do przetworzenia dla danego pliku PDF.

    Args:
        pdf_file_name (str): Name of the PDF file. / Nazwa pliku PDF.
        total_pages (int): Total number of pages in the PDF file. / Całkowita liczba stron w pliku PDF.

    Returns:
        tuple: A tuple (start_page, end_page) with the selected range.
               Krotka (start_page, end_page) z wybranym zakresem.
    """
    while True:
        choice = input(f"For file '{pdf_file_name}' (total pages: {total_pages}):\n"
                       f"Dla pliku '{pdf_file_name}' (łącznie stron: {total_pages}):\n"
                       f"1. Process all pages / Przetwórz wszystkie strony\n"
                       f"2. Specify page range (e.g., 10-20) / Podaj zakres stron (np. 10-20)\n"
                       f"Choose an option (1/2): / Wybierz opcję (1/2): ").strip()

        if choice == '1':
            return 1, total_pages
        elif choice == '2':
            page_range_str = input("Enter page range (e.g., 10-20): / Podaj zakres stron (np. 10-20): ").strip()
            try:
                start_str, end_str = page_range_str.split('-')
                start_page = int(start_str)
                end_page = int(end_str)
                if 1 <= start_page <= end_page <= total_pages:
                    return start_page, end_page
                else:
                    print(f"ERROR: Invalid page range. Ensure that {1} <= start <= end <= {total_pages}.")
                    print(
                        f"BŁĄD: Nieprawidłowy zakres stron. Upewnij się, że {1} <= początek <= koniec <= {total_pages}.")
            except ValueError:
                print("ERROR: Invalid format. Use 'START-END' format, e.g., '10-20'.")
                print("BŁĄD: Nieprawidłowy format. Użyj formatu 'START-KONIEC', np. '10-20'.")
        else:
            print("Invalid choice. Please select 1 or 2.")
            print("Nieprawidłowy wybór. Proszę wybrać 1 lub 2.")


# Function to save text to PDF with titles / Funkcja do zapisywania tekstu do PDF z tytułami
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
    print(f"Saving combined text to PDF: {output_pdf_path} with font '{font_name}' {font_size}pt...")
    print(f"Zapisywanie połączonego tekstu do PDF: {output_pdf_path} czcionką '{font_name}' {font_size}pkt...")
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        c.setFont(font_name, font_size)
        width, height = A4

        left_margin = 50  # Left margin / Lewy margines
        top_margin = height - 50  # Top margin / Górny margines
        line_height = font_size * 1.4  # Larger spacing for readability / Większy odstęp dla czytelności

        y_position = top_margin

        # Splitting content into lines, considering paragraph breaks
        # Dzielenie zawartości na linie, z uwzględnieniem podziału na akapity
        lines = []
        for paragraph in text_content.split('\n'):
            wrapped_lines = []
            if paragraph.strip():
                # Using pdfmetrics.stringWidth for precise text wrapping
                # Używamy pdfmetrics.stringWidth do precyzyjnego zawijania
                current_line = ""
                words = paragraph.split(' ')
                for word in words:
                    # Check if adding the word fits in the current line
                    # Sprawdź, czy dodanie słowa zmieści się w linii
                    if pdfmetrics.stringWidth(current_line + word + ' ', font_name, font_size) < (
                            width - 2 * left_margin):
                        current_line += word + ' '
                    else:
                        # If not, add the current line and start a new one
                        # Jeśli nie, dodaj bieżącą linię i rozpocznij nową
                        wrapped_lines.append(current_line.strip())
                        current_line = word + ' '
                if current_line.strip():  # Add any remaining words in the current line
                    # Dodaj pozostałe słowa w bieżącej linii
                    wrapped_lines.append(current_line.strip())
            else:
                wrapped_lines.append("")  # Preserve empty lines for paragraph spacing
                # Zachowaj puste linie dla odstępów między akapitami
            lines.extend(wrapped_lines)

        for line in lines:
            if y_position < 50:  # If reaching the bottom margin, create a new page
                # Jeśli dochodzimy do dolnego marginesu, nowa strona
                c.showPage()
                c.setFont(font_name, font_size)
                y_position = top_margin

            c.drawString(left_margin, y_position, line)
            y_position -= line_height

        c.save()
        print(f"Successfully saved combined PDF to: {output_pdf_path}")
        print(f"Pomyślnie zapisano połączony PDF do: {output_pdf_path}")
    except Exception as e:
        print(f"ERROR: Could not save text to PDF '{output_pdf_path}': {e}")
        print(f"BŁĄD: Nie można zapisać tekstu do PDF '{output_pdf_path}': {e}")


def merge_pdfs_with_titles():
    """
    Main function to merge PDF files from the source folder.
    It interactively prompts the user for page selection for each file,
    then combines the extracted text into a single PDF document.
    Główna funkcja do łączenia plików PDF z folderu źródłowego.
    Interaktywnie prosi użytkownika o wybór stron dla każdego pliku,
    a następnie łączy wyekstrahowany tekst w jeden dokument PDF.
    """
    print(f"Starting extraction and merging of PDF files from folder: {PDF_SOURCE_FOLDER}")
    print(f"Rozpoczynam ekstrakcję i łączenie plików PDF z folderu: {PDF_SOURCE_FOLDER}")

    # Get a list of PDF files from the source folder, sorted alphabetically
    # Pobierz listę plików PDF z folderu źródłowego, posortowanych alfabetycznie
    pdf_files = sorted([f for f in os.listdir(PDF_SOURCE_FOLDER) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print("INFO: No PDF files found in the source folder for processing.")
        print("INFO: Brak plików PDF w folderze źródłowym do przetworzenia.")
        return

    combined_text = []  # List to store extracted text from all files
    # Lista do przechowywania wyekstrahowanego tekstu z wszystkich plików

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE_FOLDER, pdf_file)
        file_title = os.path.splitext(pdf_file)[0]  # File title without extension
        # Tytuł pliku bez rozszerzenia

        print(f"\n--- Processing file: {pdf_file} ---")
        print(f"\n--- Przetwarzanie pliku: {pdf_file} ---")

        num_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
        except Exception as e:
            print(f"ERROR: Could not open PDF file '{pdf_file}' to check page count: {e}. Skipping file.")
            print(
                f"BŁĄD: Nie można otworzyć pliku PDF '{pdf_file}' w celu sprawdzenia liczby stron: {e}. Pomijam plik.")
            combined_text.append(f"\n\n--- FILE SKIPPED (PDF OPEN ERROR): {file_title} ---\n\n")
            combined_text.append(f"\n\n--- PLIK POMINIĘTY (BŁĄD OTWARCIA PDF): {file_title} ---\n\n")
            continue

        start_p, end_p = get_page_range_input(pdf_file, num_pages)  # Get page range from user
        # Pobierz zakres stron od użytkownika
        extracted_text = extract_selected_pages_from_pdf(pdf_path, start_p, end_p)  # Extract text
        # Wyekstrahuj tekst

        if extracted_text is not None:
            # Add file title as a separator before and after its content
            # Dodaj tytuł pliku jako separator przed i po jego zawartości
            combined_text.append(f"\n\n--- BEGINNING OF FILE: {file_title} ---\n\n")
            combined_text.append(f"\n\n--- POCZĄTEK PLIKU: {file_title} ---\n\n")
            combined_text.append(extracted_text)
            combined_text.append(f"\n\n--- END OF FILE: {file_title} ---\n\n")
            combined_text.append(f"\n\n--- KONIEC PLIKU: {file_title} ---\n\n")
        else:
            print(f"WARNING: Could not extract text from '{pdf_file}'. Skipping this file.")
            print(f"OSTRZEŻENIE: Nie można wyodrębnić tekstu z '{pdf_file}'. Pomijam ten plik.")
            combined_text.append(f"\n\n--- FILE SKIPPED (EXTRACTION ERROR): {file_title} ---\n\n")
            combined_text.append(f"\n\n--- PLIK POMINIĘTY (BŁĄD EKSTRAKCJI): {file_title} ---\n\n")

    if combined_text:
        final_combined_text = "\n".join(combined_text)  # Join all text fragments into a single string
        # Połącz wszystkie fragmenty tekstu w jeden string
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf_path = os.path.join(OUTPUT_FOR_ANALYSIS_FOLDER, f"Combined_Legal_Documents_{timestamp}.pdf")

        # Save the combined text to a PDF file
        # Zapisz połączony tekst do pliku PDF
        save_text_to_pdf_with_titles(final_combined_text, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
        print(f"\n--- Finished merging all PDF files to: {output_pdf_path} ---")
        print(f"\n--- Zakończono łączenie wszystkich plików PDF do: {output_pdf_path} ---")
    else:
        print("\nNo text extracted due to errors or missing files.")
        print("\nNie wyodrębniono żadnego tekstu ze względu na błędy lub brak plików.")


if __name__ == "__main__":
    # Entry point of the program. Calls the PDF merging function.
    # Punkt wejścia programu. Wywołuje funkcję łączenia PDF-ów.
    merge_pdfs_with_titles()
