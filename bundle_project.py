import os

# --- CONFIGURATION ---
# --- KONFIGURACJA ---

# The name of the output file that will contain the entire codebase.
# Nazwa pliku wyjściowego, który będzie zawierał całą bazę kodu.
OUTPUT_FILE = "codebase_bundle.txt"

# The root directory of your project. "." means the current directory where the script is located.
# Główny katalog Twojego projektu. "." oznacza bieżący katalog, w którym znajduje się skrypt.
ROOT_DIR = "."

# A list of file extensions to include in the bundle.
# Lista rozszerzeń plików, które mają zostać dołączone do paczki.
INCLUDE_EXTENSIONS = [
    ".py",
    ".yaml",
    ".yml",
    #".json",
    ".md",
    ".txt",
    ".sh"
]

# A list of directories and files to exclude from the bundle.
# Lista katalogów i plików do wykluczenia z paczki.
# I've pre-filled this based on your project structure.
# Wypełniłem tę listę na podstawie struktury Twojego projektu.
EXCLUDE_PATTERNS = [
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    "image_input",
    "image_logs",
    "image_output",
    "audio_input",
    "audio_logs",
    "audio_output",
    "LOGS",
    "PROCESSED_OUTPUT",
    "TEMP",
    "FOR_ANALYSIS",
    "TRANSLATIONS",
    OUTPUT_FILE,  # Exclude the script's own output file. / Wyklucz plik wyjściowy tego skryptu.
    "bundle_project.py",  # Exclude this script itself. / Wyklucz ten skrypt.
    ".json",

]


def bundle_codebase():
    """
    Scans the project directory, filters files, and bundles their content into a single text file.
    Skanuje katalog projektu, filtruje pliki i łączy ich zawartość w jeden plik tekstowy.
    """
    print("--- Starting project bundling ---")
    print("--- Rozpoczynam pakowanie projektu ---")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as bundle_file:
        # Write a header with project information.
        # Zapisz nagłówek z informacjami o projekcie.
        bundle_file.write(f"Project Bundle: {os.path.abspath(ROOT_DIR)}\n")
        bundle_file.write("========================================\n\n")

        # Walk through the directory structure.
        # Przejdź przez strukturę katalogów.
        for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
            # --- Exclusion Logic ---
            # --- Logika Wykluczeń ---

            # Remove excluded directories from traversal. This is efficient.
            # Usuń wykluczone katalogi z listy do przejścia. To jest wydajne.
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PATTERNS]

            # Skip any directory path that is in the exclude list.
            # Pomiń każdą ścieżkę katalogu, która jest na liście wykluczeń.
            if any(excluded in dirpath for excluded in EXCLUDE_PATTERNS):
                continue

            for filename in filenames:
                # Check if the file extension is in our include list.
                # Sprawdź, czy rozszerzenie pliku znajduje się na naszej liście dołączania.
                if any(filename.endswith(ext) for ext in INCLUDE_EXTENSIONS):
                    file_path = os.path.join(dirpath, filename)

                    # Check again for file-specific exclusion.
                    # Sprawdź ponownie pod kątem wykluczenia konkretnych plików.
                    if filename in EXCLUDE_PATTERNS:
                        continue

                    # Write file header to the bundle.
                    # Zapisz nagłówek pliku do paczki.
                    relative_path = os.path.relpath(file_path, ROOT_DIR)
                    bundle_file.write(f"--- START FILE: {relative_path} ---\n")

                    try:
                        # Write file content.
                        # Zapisz zawartość pliku.
                        with open(file_path, "r", encoding="utf-8") as file_content:
                            bundle_file.write(file_content.read())
                    except Exception as e:
                        # If reading fails, write an error message instead of content.
                        # Jeśli odczyt się nie powiedzie, zapisz komunikat o błędzie zamiast zawartości.
                        bundle_file.write(f"[Could not read file: {e}]")

                    # Write file footer.
                    # Zapisz stopkę pliku.
                    bundle_file.write(f"\n--- END FILE: {relative_path} ---\n\n")

    print(f"\n--- Project bundled successfully into '{OUTPUT_FILE}' ---")
    print(f"--- Projekt został pomyślnie spakowany do pliku '{OUTPUT_FILE}' ---")
    print("\nIMPORTANT: Please review the file for any sensitive data before sharing it.")
    print("WAŻNE: Przed udostępnieniem pliku przejrzyj go pod kątem danych wrażliwych.")


# --- Run the script ---
# --- Uruchom skrypt ---
if __name__ == "__main__":
    bundle_codebase()