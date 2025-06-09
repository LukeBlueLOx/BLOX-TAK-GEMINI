import os
import pdfplumber
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Konfiguracja ścieżek ---
# Folder źródłowy dla plików PDF
PDF_SOURCE_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/TEMP"
# Folder docelowy dla wynikowego pliku PDF
OUTPUT_FOR_ANALYSIS_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/FOR_ANALYSIS"

# Zdefiniuj ścieżkę do pliku czcionki Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-R.ttf' znajduje się w tej samej lokalizacji lub podaj pełną ścieżkę.
UBUNTU_MONO_FONT_PATH = os.path.join(os.path.dirname(__file__), "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf")

# Utwórz foldery, jeśli nie istnieją
os.makedirs(PDF_SOURCE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOR_ANALYSIS_FOLDER, exist_ok=True)

# Zarejestruj czcionkę Ubuntu Mono w ReportLab.
try:
    pdfmetrics.registerFont(TTFont('UbuntuMono', UBUNTU_MONO_FONT_PATH))
    print(f"Czcionka 'UbuntuMono' zarejestrowana pomyślnie z {UBUNTU_MONO_FONT_PATH}.")
    DEFAULT_FONT = "UbuntuMono"
except Exception as e:
    print(f"BŁĄD: Nie można zarejestrować czcionki Ubuntu Mono z {UBUNTU_MONO_FONT_PATH}. Upewnij się, że plik istnieje i jest dostępny. Błąd: {e}")
    DEFAULT_FONT = "Helvetica" # Fallback do domyślnej czcionki

FONT_SIZE = 12

# --- Funkcja do ekstrakcji tekstu z WYBRANYCH STRON ---
def extract_selected_pages_from_pdf(pdf_path, start_page, end_page):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)

            actual_start_index = max(0, start_page - 1)
            actual_end_index = min(num_pages, end_page)

            if actual_start_index >= num_pages:
                print(f"OSTRZEŻENIE: Strona początkowa {start_page} wykracza poza liczbę stron PDF ({num_pages}). Zwracam pusty tekst.")
                return ""
            if actual_start_index >= actual_end_index:
                print(f"OSTRZEŻENIE: Zakres stron ({start_page}-{end_page}) jest nieprawidłowy lub pusty. Zwracam pusty tekst.")
                return ""

            print(f"Ekstrakcja stron od {actual_start_index + 1} do {actual_end_index} z pliku '{os.path.basename(pdf_path)}'...")

            for i in range(actual_start_index, actual_end_index):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    print(f"  INFO: Strona {i + 1} jest pusta lub nie zawiera tekstu.")
    except Exception as e:
        print(f"BŁĄD: Nie można wyodrębnić tekstu z {pdf_path} dla stron {start_page}-{end_page}: {e}")
        return None
    return text.strip()

# --- Funkcja do interaktywnego pobierania zakresu stron ---
def get_page_range_input(pdf_file_name, total_pages):
    while True:
        choice = input(f"Dla pliku '{pdf_file_name}' (łącznie stron: {total_pages}):\n"
                       f"1. Przetwórz wszystkie strony\n"
                       f"2. Podaj zakres stron (np. 10-20)\n"
                       f"Wybierz opcję (1/2): ").strip()

        if choice == '1':
            return 1, total_pages
        elif choice == '2':
            page_range_str = input("Podaj zakres stron (np. 10-20): ").strip()
            try:
                start_str, end_str = page_range_str.split('-')
                start_page = int(start_str)
                end_page = int(end_str)
                if 1 <= start_page <= end_page <= total_pages:
                    return start_page, end_page
                else:
                    print(f"BŁĄD: Nieprawidłowy zakres stron. Upewnij się, że {1} <= początek <= koniec <= {total_pages}.")
            except ValueError:
                print("BŁĄD: Nieprawidłowy format. Użyj formatu 'START-KONIEC', np. '10-20'.")
        else:
            print("Nieprawidłowy wybór. Proszę wybrać 1 lub 2.")

# Funkcja do zapisywania tekstu do PDF
def save_text_to_pdf_with_titles(text_content, output_pdf_path, font_name, font_size):
    print(f"Zapisywanie połączonego tekstu do PDF: {output_pdf_path} czcionką '{font_name}' {font_size}pkt...")
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        c.setFont(font_name, font_size)
        width, height = A4

        left_margin = 50
        top_margin = height - 50
        line_height = font_size * 1.4  # Większy odstęp dla czytelności

        y_position = top_margin

        # Dzielenie zawartości na linie, z uwzględnieniem podziału na akapity
        lines = []
        for paragraph in text_content.split('\n'):
            wrapped_lines = []
            if paragraph.strip():
                # Używamy pdfmetrics.stringWidth do precyzyjnego zawijania
                current_line = ""
                words = paragraph.split(' ')
                for word in words:
                    # Sprawdź, czy dodanie słowa zmieści się w linii
                    if pdfmetrics.stringWidth(current_line + word + ' ', font_name, font_size) < (width - 2 * left_margin):
                        current_line += word + ' '
                    else:
                        # Jeśli nie, dodaj bieżącą linię i rozpocznij nową
                        wrapped_lines.append(current_line.strip())
                        current_line = word + ' '
                if current_line.strip(): # Dodaj pozostałe słowa w bieżącej linii
                    wrapped_lines.append(current_line.strip())
            else:
                wrapped_lines.append("") # Zachowaj puste linie dla odstępów między akapitami
            lines.extend(wrapped_lines)

        for line in lines:
            if y_position < 50: # Jeśli dochodzimy do dolnego marginesu, nowa strona
                c.showPage()
                c.setFont(font_name, font_size)
                y_position = top_margin

            c.drawString(left_margin, y_position, line)
            y_position -= line_height

        c.save()
        print(f"Pomyślnie zapisano połączony PDF do: {output_pdf_path}")
    except Exception as e:
        print(f"BŁĄD: Nie można zapisać tekstu do PDF '{output_pdf_path}': {e}")

def merge_pdfs_with_titles():
    print(f"Rozpoczynam ekstrakcję i łączenie plików PDF z folderu: {PDF_SOURCE_FOLDER}")

    pdf_files = sorted([f for f in os.listdir(PDF_SOURCE_FOLDER) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print("INFO: Brak plików PDF w folderze źródłowym do przetworzenia.")
        return

    combined_text = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE_FOLDER, pdf_file)
        file_title = os.path.splitext(pdf_file)[0] # Tytuł pliku bez rozszerzenia

        print(f"\n--- Przetwarzanie pliku: {pdf_file} ---")

        num_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
        except Exception as e:
            print(f"BŁĄD: Nie można otworzyć pliku PDF '{pdf_file}' w celu sprawdzenia liczby stron: {e}. Pomijam plik.")
            combined_text.append(f"\n\n--- PLIK POMINIĘTY (BŁĄD OTWARCIA PDF): {file_title} ---\n\n")
            continue

        start_p, end_p = get_page_range_input(pdf_file, num_pages)
        extracted_text = extract_selected_pages_from_pdf(pdf_path, start_p, end_p)

        if extracted_text is not None:
            # Dodaj tytuł pliku jako separator
            combined_text.append(f"\n\n--- POCZĄTEK PLIKU: {file_title} ---\n\n")
            combined_text.append(extracted_text)
            combined_text.append(f"\n\n--- KONIEC PLIKU: {file_title} ---\n\n")
        else:
            print(f"OSTRZEŻENIE: Nie można wyodrębnić tekstu z '{pdf_file}'. Pomijam ten plik.")
            combined_text.append(f"\n\n--- PLIK POMINIĘTY (BŁĄD EKSTRAKCJI): {file_title} ---\n\n")

    if combined_text:
        final_combined_text = "\n".join(combined_text)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf_path = os.path.join(OUTPUT_FOR_ANALYSIS_FOLDER, f"Combined_Legal_Documents_{timestamp}.pdf")
        save_text_to_pdf_with_titles(final_combined_text, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
        print(f"\n--- Zakończono łączenie wszystkich plików PDF do: {output_pdf_path} ---")
    else:
        print("\nNie wyodrębniono żadnego tekstu ze względu na błędy lub brak plików.")

if __name__ == "__main__":
    merge_pdfs_with_titles()
