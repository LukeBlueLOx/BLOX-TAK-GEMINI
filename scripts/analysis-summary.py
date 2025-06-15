import os
import pdfplumber
import google.generativeai as genai
import textwrap
import datetime
import json  # Import needed for log files
import yaml

# --- PDF Generation Libraries ---
# --- Biblioteki do generowania PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

with open("config.yaml", "r") as cr:
    config_vals = yaml.full_load(cr)
KEY = config_vals['KEY']

# --- Path Configuration ---
# --- Konfiguracja ścieżek ---
PDF_INPUT_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/FOR_ANALYSIS"
OUTPUT_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/PROCESSED_OUTPUT"
LOG_FOLDER = "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/LOGS"

# Aktualny czas
now = datetime.datetime.now()

# Format ISO 8601 z milisekundami
full_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

# Define the path to the Ubuntu Mono font file.
# Make sure 'UbuntuMono-R.ttf' is in the same directory as this script, or provide a full path.
# Zdefiniuj ścieżkę do pliku czcionki Ubuntu Mono.
# Upewnij się, że 'UbuntuMono-R.ttf' znajduje się w tym samym katalogu co ten skrypt, lub podaj pełną ścieżkę.
UBUNTU_MONO_FONT_PATH = os.path.join(os.path.dirname(__file__), "/home/luke_blue_lox/PycharmProjects/BLOX-TAK-GEMINI/UbuntuMono-Regular.ttf")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

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
GOOGLE_API_KEY = KEY
genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully.")
print("Gemini API skonfigurowane pomyślnie.")

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

CHUNK_SIZE = 100_000  # Rozmiar chunka dla pojedynczych dokumentów
OVERALL_SUMMARY_CHUNK_SIZE = 50_000  # Rozmiar chunka dla globalnego podsumowania - można dostosować


# --- Funkcja do ekstrakcji CAŁEGO tekstu ---
def extract_text_from_pdf_full(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"BŁĄD: Nie można wyodrębnić tekstu z {pdf_path}: {e}")
        return None
    return text.strip()


# --- Funkcja do ekstrakcji tekstu z WYBRANYCH STRON ---
def extract_selected_pages_from_pdf(pdf_path, start_page, end_page):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)

            actual_start_index = max(0, start_page - 1)
            actual_end_index = min(num_pages, end_page)

            if actual_start_index >= num_pages:
                print(
                    f"OSTRZEŻENIE: Strona początkowa {start_page} wykracza poza liczbę stron PDF ({num_pages}). Zwracam pusty tekst.")
                return ""
            if actual_start_index >= actual_end_index:
                print(
                    f"OSTRZEŻENIE: Zakres stron ({start_page}-{end_page}) jest nieprawidłowy lub pusty. Zwracam pusty tekst.")
                return ""

            print(
                f"Ekstrakcja stron od {actual_start_index + 1} do {actual_end_index} z pliku '{os.path.basename(pdf_path)}'...")

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
                    print(
                        f"BŁĄD: Nieprawidłowy zakres stron. Upewnij się, że {1} <= początek <= koniec <= {total_pages}.")
            except ValueError:
                print("BŁĄD: Nieprawidłowy format. Użyj formatu 'START-KONIEC', np. '10-20'.")
        else:
            print("Nieprawidłowy wybór. Proszę wybrać 1 lub 2.")


def analyze_text_with_gemini(text_to_analyze, prompt_prefix=""):
    if not text_to_analyze.strip():
        print("INFO: Brak tekstu do analizy. Zwracam pusty string.")
        return "", 0, 0

    chunks = textwrap.wrap(text_to_analyze, CHUNK_SIZE, break_long_words=False, replace_whitespace=False)

    if not chunks:
        print("OSTRZEŻENIE: Tekst nie został podzielony na chunki prawidłowo. Zwracam pusty string.")
        return "", 0, 0

    full_analysis = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Tekst zostanie podzielony na {len(chunks)} części do analizy.")

    for i, chunk in enumerate(chunks):
        print(f"Analizuję część {i + 1}/{len(chunks)} ({len(chunk)} znaków)...")

        base_prompt = (
                f"Jesteś wysoce doświadczonym ekspertem prawnym, specjalizującym się w prawie cywilnym, egzekucyjnym i socjalnym w Polsce. "
                f"Przeanalizuj dokument pod kątem Prawa Polskiego i Unii Europejskiej, zidentyfikuj kluczowe fakty prawne, terminy, strony, roszczenia, zobowiązania, dowody oraz oświadczenia dotyczące sytuacji finansowej i zdrowotnej Łukasza Andruszkiewicza. "
                f"Szczególną uwagę zwróć na: odniesienia do sytuacji finansowej, długów, dochodów, zatrudnienia, prób znalezienia pracy; szczegóły stanu zdrowia, wypadków, urazów, braku ubezpieczenia; wzmianki o próbach uzyskania pomocy od instytucji; konieczność podjęcia pracy zdalnej; oświadczenia dotyczące braku majątku i trudności egzystencji; adresatów, daty i sygnatury akt. Zacytuj odpowiednie artykuły i opisz jakie prawa zostały złamane. "
                f"Wynik podaj w języku Polskim i Angielskim. Podaj również na końcu treść użytego promptu - analogicznie w języku "
                f"Polskim i Angielskim, a także wersję modelu jaki został użyty - czyli: gemini-1.5-flash,"
                f"z pełnym timestamp: " + full_timestamp
        )
        prompt = f"{prompt_prefix}\n\n{base_prompt}\n\nTekst do analizy:\n\n{chunk}"

        current_chunk_input_tokens = 0
        current_chunk_output_tokens = 0
        chunk_analysis_part = ""

        try:
            current_chunk_input_tokens = model.count_tokens(prompt).total_tokens
            total_input_tokens += current_chunk_input_tokens
            print(f"  Szacowane tokeny wejściowe dla tej części: {current_chunk_input_tokens}")

            response = model.generate_content(prompt)

            if response.parts:
                chunk_analysis_part = "".join([part.text for part in response.parts]).strip()
                full_analysis.append(chunk_analysis_part)

                try:
                    if hasattr(response, '_result') and 'usageMetadata' in response._result:
                        usage_metadata = response._result['usageMetadata']
                        current_chunk_output_tokens = usage_metadata.get('candidatesTokenCount', 0)
                        print(
                            f"  Wygenerowane tokeny wyjściowe dla tej części (z usageMetadata): {current_chunk_output_tokens}")
                    else:
                        print(
                            "  Brak usageMetadata w odpowiedzi, mimo obecności treści. Oszacowanie tokenów wyjściowych na podstawie tekstu.")
                        current_chunk_output_tokens = model.count_tokens(chunk_analysis_part).total_tokens
                        print(f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {current_chunk_output_tokens}")
                except Exception as usage_e:
                    print(f"BŁĄD podczas pobierania usageMetadata dla części {i + 1}: {usage_e}")
                    print("  Oszacowanie tokenów wyjściowych na podstawie wygenerowanego tekstu.")
                    current_chunk_output_tokens = model.count_tokens(chunk_analysis_part).total_tokens
                    print(f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {current_chunk_output_tokens}")
            else:
                print(
                    f"OSTRZEŻENIE: Gemini nie zwróciło żadnego tekstu (response.parts jest puste) dla części {i + 1}.")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"  Feedback od Gemini (Prompt Feedback): {response.prompt_feedback}")
                    full_analysis.append(f"[PROBLEM: FEEDBACK OD GEMINI - {response.prompt_feedback}]\n")
                if hasattr(response, 'candidates') and response.candidates:
                    print("  Dostępne kandydujące odpowiedzi (może zawierać powody blokady):")
                    for candidate in response.candidates:
                        print(f"    Finish Reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            print(f"    Safety Ratings: {candidate.safety_ratings}")
                            full_analysis.append(f"[PROBLEM: BEZPIECZEŃSTWO - {candidate.safety_ratings}]\n")
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            print(f"    Content parts (puste?): {candidate.content.parts}")
                else:
                    print("  Brak 'candidates' w odpowiedzi (coś poszło bardzo źle).")
                full_analysis.append(f"[BRAK ANALIZY DLA TEJ CZĘŚCI: PROBLEM Z API LUB TREŚCIĄ]\n")

            total_output_tokens += current_chunk_output_tokens

        except genai.types.BlockedPromptException as e:
            print(f"BŁĄD: Żądanie zablokowane przez Gemini (safety settings) dla części {i + 1}: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"  Szczegóły błędu z API Gemini: {e.response.text}")
                full_analysis.append(f"[ANALIZA ZABLOKOWANA PRZEZ API: {e.response.text}]\n")
            else:
                full_analysis.append(f"[ANALIZA ZABLOKOWANA ZE WZGLĘDÓW BEZPIECZEŃSTWA - BLOCKED PROMPT EXCEPTION]\n")
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini dla części {i + 1}: {e}")
            print(f"  Typ błędu: {type(e).__name__}")
            if hasattr(e, 'response') and e.response:
                print(f"  Szczegóły błędu z API Gemini: {e.response.text}")
            full_analysis.append(f"[BŁĄD ANALIZY DLA TEJ CZĘŚCI: {e}]\n")

    final_analysis = "\n\n---\n\n".join(full_analysis)
    return final_analysis, total_input_tokens, total_output_tokens


# --- Nowa funkcja do globalnego podsumowania ---
def summarize_overall_legal_findings(all_summaries_text):
    print("\n\n--- Generowanie globalnego podsumowania prawnego ---")
    if not all_summaries_text.strip():
        print("Brak danych do globalnego podsumowania. Zwracam pusty tekst.")
        return "", 0, 0

    overall_chunks = textwrap.wrap(all_summaries_text, OVERALL_SUMMARY_CHUNK_SIZE, break_long_words=False,
                                   replace_whitespace=False)

    overall_summary_parts = []
    overall_input_tokens = 0
    overall_output_tokens = 0

    base_overall_prompt = (
        f"Jesteś wysoce doświadczonym ekspertem prawnym, specjalizującym się w prawie cywilnym, egzekucyjnym i socjalnym w Polsce. "
        f"Twoim zadaniem jest przygotowanie kompleksowych i rzeczowych wyjaśnień dla Kancelarii Komorniczych, "
        f"bazując na całości dostarczonego dokumentu (połączonego z wielu źródeł). "
        f"Celem jest przedstawienie aktualnej, bardzo trudnej sytuacji finansowej, zdrowotnej i życiowej dłużnika Łukasza Andruszkiewicza, "
        f"zgodnie z wezwaniem od Komornika Joanny Majdak (odwołując się do treści dokumentu). "
        f"W swoich wyjaśnieniach, uwzględnij i szczegółowo opisz następujące punkty, odwołując się do treści dokumentu i przekazanych informacji:"
        f"\n\n1.  **Potwierdzenie aktualnej trudnej sytuacji:** Jasno zaznacz, że sytuacja dłużnika nie uległa poprawie od poprzednich wyjaśnień i jest bardzo trudna (odwołaj się do dokumentów np. 'SPRZECIW_Nc-e_1932318_24_2025-04-27.pdf' oraz 'Gmail - Wyjaśnienia 2024-08-12.PDF', wskazując na pogorszenie i brak możliwości spłaty). "
        f"\n\n2.  **Szczegółowy opis stanu zdrowia i jego wpływu na sytuację:** "
        f"    Wspomnij o wypadku z 1 maja, nieleczonych urazach (oczodół, policzek, zęby, staw skroniowo-żuchwowy, kolano, ścięgno Achillesa), braku ubezpieczenia zdrowotnego i niemożności odbycia badań kontrolnych (odwołaj się do 'Gmail - KM 1623_22.pdf' oraz 'Cover Letter.pdf'). "
        f"    Podkreśl zagrożenie neuralgią nerwu trójdzielnego jako konsekwencję urazów ('SPRZECIW_Nc-e_1932318_24_2025-04-27.pdf'). "
        f"    Wspomnij o braku wsparcia instytucjonalnego w kwestii zdrowia i ubezpieczenia (np. odmowa PUP). "
        f"\n\n3.  **Opis sytuacji finansowej i majątkowej:** "
        f"    Zadeklaruj brak możliwości spłaty zadłużenia. "
        f"    Jasno określ, że jedyną rzeczą, jaką udało się nabyć, są przedmioty z faktury 'F_2025_19384_1.pdf' (adapter dysku NVME M.2, Raspberry Pi 256GB SSD, koszt dostawy), podając ich wartość. "
        f"    Wyjaśnij pochodzenie środków na ten zakup: ostatnie odłożone pieniądze od zeszłego roku, w tym 100 PLN od Brata na święta i 100 PLN od rodziców za rozliczenie zeznań podatkowych w bieżącym roku. "
        f"    Podkreśl, że był to **konieczny wydatek** w celu dokończenia portfolio związanego z ekosystemem TAK, co jest kluczowe dla prób zarobienia pieniędzy. "
        f"    Wspomnij o aktywnych, lecz dotychczas bezskutecznych próbach znalezienia zatrudnienia/współpracy, co prowadzi do braku dochodów. "
        f"    Potwierdź, że dłużnik nie posiada innych znaczących środków ani majątku poza wymienionymi. "
        f"    Odwołaj się do wszelkich wcześniejszych oświadczeń o trudnej sytuacji finansowej, wyzysku, braku wsparcia od Państwa Polskiego i UE, oraz żądaniach odszkodowań (np. w 'SPRZECIW_2025-03-03.pdf', 'ODWOŁANIE-SeriaP_Nr0360-2025-01-13_GOV-PL_2025-01-30'). "
        f"\n\n4.  **Konsekwencje i oczekiwania dłużnika:** "
        f"    Wspomnij o konieczności prowadzenia korespondencji z zagranicy i braku odpowiedzi. "
        f"    Zaznacz, że dłużnik żąda prawnika, którego wynagrodzenie pokryje Fundusz Sprawiedliwości, oraz renty czasowej z ubezpieczeniem zdrowotnym, aby mógł zadbać o swoje zdrowie i przeprowadzić upadłość. ('SPRZECIW_2025-03-03.pdf') "
        f"    Podkreśl, że dłużnik nie jest w stanie obecnie stawiać się przed instytucjami w regionie (Dolny Śląsk) ze względu na doświadczenia (odwołaj się do 'Pismo Właściwe.pdf' oraz 'SPRZECIW_2025-03-03.pdf')."
        f"\n\nZadbaj o to, aby wyjaśnienia były kompleksowe, spójne, rzeczowe i empatyczne, jednocześnie ściśle trzymając się faktów zawartych w dokumentacji. Tekst wygenerowany przez model będzie stanowił trzon pisma do kancelarii komorniczej."
    )

    for i, chunk in enumerate(overall_chunks):
        print(f"Analizuję część {i + 1}/{len(overall_chunks)} globalnego podsumowania ({len(chunk)} znaków)...")
        prompt = f"{base_overall_prompt}\n\nAnalizy do podsumowania:\n\n{chunk}"

        chunk_input_tokens = 0
        chunk_output_tokens = 0
        current_summary_part = ""

        try:
            chunk_input_tokens = model.count_tokens(prompt).total_tokens
            overall_input_tokens += chunk_input_tokens
            print(f"  Szacowane tokeny wejściowe dla tej części globalnego podsumowania: {chunk_input_tokens}")

            response = model.generate_content(prompt)

            if response.parts:
                current_summary_part = "".join([part.text for part in response.parts]).strip()
                overall_summary_parts.append(current_summary_part)

                try:
                    if hasattr(response, '_result') and 'usageMetadata' in response._result:
                        usage_metadata = response._result['usageMetadata']
                        chunk_output_tokens = usage_metadata.get('candidatesTokenCount', 0)
                        print(
                            f"  Wygenerowane tokeny wyjściowe dla tej części (z usageMetadata): {chunk_output_tokens}")
                    else:
                        print(
                            "  Brak usageMetadata w odpowiedzi, mimo obecności treści. Oszacowanie tokenów wyjściowych na podstawie tekstu.")
                        chunk_output_tokens = model.count_tokens(current_summary_part).total_tokens
                        print(f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {chunk_output_tokens}")
                except Exception as usage_e:
                    print(
                        f"BŁĄD podczas pobierania usageMetadata dla części globalnego podsumowania {i + 1}: {usage_e}")
                    print("  Oszacowanie tokenów wyjściowych na podstawie wygenerowanego tekstu.")
                    chunk_output_tokens = model.count_tokens(current_summary_part).total_tokens
                    print(f"  Oszacowane tokeny wyjściowe dla tej części (z tekstu): {chunk_output_tokens}")
            else:
                print(f"OSTRZEŻENIE: Gemini nie zwróciło żadnego tekstu dla globalnego podsumowania (część {i + 1}).")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"  Feedback od Gemini: {response.prompt_feedback}")
                overall_summary_parts.append(f"[BRAK PODSUMOWANIA DLA TEJ CZĘŚCI: PROBLEM Z API LUB TREŚCIĄ]\n")

            overall_output_tokens += chunk_output_tokens

        except genai.types.BlockedPromptException as e:
            print(f"BŁĄD: Globalne podsumowanie zablokowane przez Gemini (safety settings) dla części {i + 1}: {e}")
            overall_summary_parts.append(f"[GLOBALNE PODSUMOWANIE ZABLOKOWANE PRZEZ API]\n")
        except Exception as e:
            print(
                f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini dla globalnego podsumowania (część {i + 1}): {e}")
            overall_summary_parts.append(f"[GLOBALNE PODSUMOWANIE: BŁĄD ANALIZY DLA TEJ CZĘŚCI: {e}]\n")

    final_overall_summary = "\n\n".join(overall_summary_parts)
    return final_overall_summary, overall_input_tokens, overall_output_tokens


def write_usage_summary(total_files, total_input_tokens, total_output_tokens, total_duration,
                        overall_summary_input_tokens=0, overall_summary_output_tokens=0):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_FOLDER, f"gemini_usage_summary_{timestamp}.txt")

    summary_content = (
        f"--- Podsumowanie Uruchomienia BLOX-TAK-GEMINI ---\n"
        f"Data i czas uruchomienia: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Przetworzone plików PDF: {total_files}\n"
        f"Użyty model Gemini: {MODEL_NAME}\n"
        f"Całkowita liczba tokenów wejściowych (prompt - per dokument): {total_input_tokens}\n"
        f"Całkowita liczba tokenów wyjściowych (generacja - per dokument): {total_output_tokens}\n"
        f"Całkowita liczba tokenów wejściowych (prompt - globalne podsumowanie): {overall_summary_input_tokens}\n"
        f"Całkowita liczba tokenów wyjściowych (generacja - globalne podsumowanie): {overall_summary_output_tokens}\n"
        f"Łączny czas przetwarzania: {total_duration:.2f} sekund\n"
        f"--------------------------------------------------\n"
        f"UWAGA: Jeśli tokeny wyjściowe wynoszą 0 lub są podejrzanie niskie mimo wygenerowanej treści, "
        f"może to oznaczać problem z API lub blokadę treści przez filtry bezpieczeństwa Gemini, "
        f"lub błąd w pobieraniu usageMetadata. Sprawdź logi konsoli powyżej dla 'Feedback od Gemini' lub 'Safety Ratings'.\n"
    )

    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
        print(f"\n--- Podsumowanie zużycia zapisano do: {log_file_path} ---")
    except Exception as e:
        print(f"BŁĄD: Nie można zapisać pliku podsumowania zużycia: {e}")


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

        lines = []
        for paragraph in text_content.split('\n'):
            wrapped_lines = []
            if paragraph.strip():
                char_width = pdfmetrics.stringWidth('M', font_name, font_size)
                max_chars_per_line = int((width - 2 * left_margin) / char_width) if char_width > 0 else 100

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


def process_all_pdfs_with_gemini():
    print(f"Rozpoczynam analizę plików PDF z folderu: {PDF_INPUT_FOLDER}")

    pdf_files = [f for f in os.listdir(PDF_INPUT_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("INFO: Brak plików PDF w folderze do analizy.")
        return

    overall_total_input_tokens = 0
    overall_total_output_tokens = 0
    processed_files_count = 0

    legal_summaries = []
    all_individual_analyses_text = []

    start_time = datetime.datetime.now()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_INPUT_FOLDER, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_legal_analysis_{timestamp}.txt")
        output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_legal_analysis_{timestamp}.pdf")

        print(f"\n--- Przetwarzanie pliku: {pdf_file} ---")

        num_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
        except Exception as e:
            print(
                f"BŁĄD: Nie można otworzyć pliku PDF '{pdf_file}' w celu sprawdzenia liczby stron: {e}. Pomijam plik.")
            error_msg = f"[BŁĄD EKSTRAKCJI TEKSTU Z PDF: {e}]"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            save_text_to_pdf(error_msg, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
            continue

        start_p, end_p = get_page_range_input(pdf_file, num_pages)

        extracted_text = extract_selected_pages_from_pdf(pdf_path, start_p, end_p)

        if extracted_text is None:
            print(f"BŁĄZ: Pomijam plik '{pdf_file}' z powodu problemów z ekstrakcją tekstu.")
            summary_content = f"--- Analiza dla pliku: {pdf_file} ---\n[BŁĄD EKSTRAKCJI TEKSTU Z PDF]\n"
            error_msg = "[BŁĄD EKSTRAKCJI TEKSTU Z PDF]"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            save_text_to_pdf(error_msg, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
            legal_summaries.append(summary_content)
            all_individual_analyses_text.append(summary_content)
            continue

        if not extracted_text:
            print(
                f"INFO: Plik '{pdf_file}' jest pusty lub nie zawiera tekstu po ekstrakcji dla wybranego zakresu. Pomijam analizę.")
            summary_content = f"--- Analiza dla pliku: {pdf_file} ---\n[PLIK PUSTY LUB BEZ TEKSTU DO ANALIZY]\n"
            info_msg = "[PLIK PUSTY LUB BEZ TEKSTU DO ANALIZY]"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(info_msg)
            save_text_to_pdf(info_msg, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
            legal_summaries.append(summary_content)
            all_individual_analyses_text.append(summary_content)
            continue

        legal_analysis_result, file_input_tokens, file_output_tokens = analyze_text_with_gemini(extracted_text)

        overall_total_input_tokens += file_input_tokens
        overall_total_output_tokens += file_output_tokens
        processed_files_count += 1

        if legal_analysis_result is not None and legal_analysis_result.strip():
            try:
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(legal_analysis_result)
                print(f"SUKCES: Wynik analizy prawnej zapisano do TXT: {output_txt_path}")

                save_text_to_pdf(legal_analysis_result, output_pdf_path, DEFAULT_FONT, FONT_SIZE)

                summary_content = f"--- Analiza dla pliku: {pdf_file} ---\n" + legal_analysis_result + "\n"
            except Exception as e:
                print(f"BŁĄD: Nie można zapisać wyniku analizy dla {pdf_file}: {e}")
                summary_content = f"--- Analiza dla pliku: {pdf_file} ---\n[BŁĄD ZAPISU WYNIKU ANALIZY: {e}]\n"
                save_text_to_pdf(summary_content, output_pdf_path, DEFAULT_FONT, FONT_SIZE)
        else:
            print(
                f"OSTRZEŻENIE: Nie uzyskano sensownego wyniku analizy dla '{pdf_file}'. Nie zapisano pliku wyjściowego TXT.")
            summary_content = f"--- Analiza dla pliku: {pdf_file} ---\n[NIE UZYSKANO WYNIKU ANALIZY Z GEMINI]\n"
            info_msg = "[NIE UZYSKANO WYNIKU ANALIZY Z GEMINI]"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(info_msg)
            save_text_to_pdf(info_msg, output_pdf_path, DEFAULT_FONT, FONT_SIZE)

        legal_summaries.append(summary_content)
        all_individual_analyses_text.append(
            legal_analysis_result if legal_analysis_result else summary_content)

        print(f"--- Zużycie tokenów dla pliku '{pdf_file}' ---")
        print(f"  Tokeny wejściowe: {file_input_tokens}")
        print(f"  Tokeny wyjściowe: {file_output_tokens}")
        print("-------------------------------------------------")

    end_time = datetime.datetime.now()
    total_processing_duration = (end_time - start_time).total_seconds()

    print("\n--- Zakończono przetwarzanie wszystkich plików PDF. ---")

    print("\n\n--- ZBIORCZE PODSUMOWANIE PRAWNE DLA KAŻDEGO DOKUMENTU ---")
    for summary in legal_summaries:
        print(summary)
    print("----------------------------------------------------------\n")

    combined_analyses_for_overall_summary = "\n\n".join(filter(None, all_individual_analyses_text))
    overall_legal_summary, overall_summary_input_tokens, overall_summary_output_tokens = \
        summarize_overall_legal_findings(combined_analyses_for_overall_summary)

    print("\n\n--- GLOBALNE PODSUMOWANIE PRAWNE (dla wszystkich dokumentów) ---")
    if overall_legal_summary.strip():
        print(overall_legal_summary)
        global_summary_txt_file_path = os.path.join(OUTPUT_FOLDER,
                                                    f"GLOBAL_LEGAL_SUMMARY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(global_summary_txt_file_path, "w", encoding="utf-8") as f:
                f.write(overall_legal_summary)
            print(f"\nGlobalne podsumowanie zapisano do TXT: {global_summary_txt_file_path}")
        except Exception as e:
            print(f"BŁĄD: Nie można zapisać globalnego podsumowania do TXT: {e}")

        global_summary_pdf_file_path = os.path.join(OUTPUT_FOLDER,
                                                    f"GLOBAL_LEGAL_SUMMARY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        save_text_to_pdf(overall_legal_summary, global_summary_pdf_file_path, DEFAULT_FONT, FONT_SIZE)

    else:
        print("[BRAK GLOBALNEGO PODSUMOWANIA - MOŻLIWY PROBLEM Z API LUB BRAK TREŚCI]")
        save_text_to_pdf("[BRAK GLOBALNEGO PODSUMOWANIA - MOŻLIWY PROBLEM Z API LUB BRAK TREŚCI]",
                         os.path.join(OUTPUT_FOLDER,
                                      f"GLOBAL_LEGAL_SUMMARY_ERROR_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"),
                         DEFAULT_FONT, FONT_SIZE)

    print("------------------------------------------------------------------\n")

    write_usage_summary(processed_files_count, overall_total_input_tokens, overall_total_output_tokens,
                        total_processing_duration, overall_summary_input_tokens, overall_summary_output_tokens)


if __name__ == "__main__":
    process_all_pdfs_with_gemini()