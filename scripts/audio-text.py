import os
import google.generativeai as genai
import datetime
import time
import json  # Added for JSON log saving / Dodane do zapisu logów w formacie JSON
import yaml

with open("config.yaml", "r") as cr:
    config_vals = yaml.full_load(cr)
KEY = config_vals['KEY']

# --- PATH CONFIGURATION ---
# --- KONFIGURACJA ŚCIEŻEK ---
# Folder for input audio files.
# Folder na wejściowe pliki audio.
AUDIO_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "audio_input")
# Folder for output transcription and analysis results.
# Folder na wyniki transkrypcji i analizy.
AUDIO_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "audio_output")
# Folder for logs of audio processing.
# Folder na logi przetwarzania audio.
AUDIO_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "audio_logs")

# Create directories if they don't exist.
# Tworzy katalogi, jeśli nie istnieją.
os.makedirs(AUDIO_INPUT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_LOG_FOLDER, exist_ok=True)

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


# --- Functions for Audio File Handling and Gemini Audio-to-Text ---
# --- Funkcje do obsługi plików audio i Gemini Audio-to-Text ---

def upload_audio_to_gemini_files_api(audio_file_path):
    # Print message indicating file upload.
    # Wyświetla komunikat o przesyłaniu pliku.
    print(f"Uploading audio file '{os.path.basename(audio_file_path)}' to Gemini Files API...")
    print(f"Przesyłanie pliku audio '{os.path.basename(audio_file_path)}' do Gemini Files API...")
    try:
        # Upload the file using genai.upload_file.
        # Przesyła plik za pomocą genai.upload_file.
        file = genai.upload_file(path=audio_file_path)
        print(f"File '{os.path.basename(audio_file_path)}' uploaded. Gemini File Name: {file.name}")
        print(f"Plik '{os.path.basename(audio_file_path)}' przesłany. Nazwa pliku Gemini: {file.name}")
        return file
    except Exception as e:
        # Error message if upload fails.
        # Komunikat o błędzie, jeśli przesłanie pliku się nie powiedzie.
        print(f"ERROR: Could not upload file '{audio_file_path}' to Gemini Files API: {e}")
        print(f"BŁĄD: Nie można przesłać pliku '{audio_file_path}' do Gemini Files API: {e}")
        return None


def transcribe_audio_with_gemini_model(audio_file_object, model_name, log_data):
    # Print message indicating audio transcription.
    # Wyświetla komunikat o transkrypcji audio.
    print(f"Transcribing audio using Gemini model '{model_name}'...")
    print(f"Transkrypcja audio za pomocą modelu Gemini '{model_name}'...")

    # Initialize the GenerativeModel.
    # Inicjuje model GenerativeModel.
    model = genai.GenerativeModel(model_name=model_name)

    # Prompt for transcription and stylistic/grammatical correction in Polish.
    # Prompt do transkrypcji i poprawy stylistycznej/gramatycznej w języku polskim.
    prompt_parts = [
        "Transcribe the audio recording into text. Return the text stylistically and grammatically corrected in Polish."
        "Przetranskrybuj nagranie audio na tekst. Zwróć tekst poprawiony stylistycznie i gramatycznie w języku polskim."
    ]

    # Count tokens for input (audio file and prompt).
    # Liczy tokeny dla inputu (pliku audio i promptu).
    try:
        # Note: count_tokens accepts a list of objects, including the audio_file_object and prompt part.
        # Uwaga: count_tokens przyjmuje listę obiektów, w tym audio_file_object i część promptu.
        count_response = model.count_tokens([audio_file_object, prompt_parts[0]])
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
        # Pass the audio and prompt to generate_content.
        # Przekazuje audio i prompt do generate_content.
        response = model.generate_content([audio_file_object, prompt_parts[0]])
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
        # Error message if audio transcription/analysis fails.
        # Komunikat o błędzie, jeśli transkrypcja/analiza audio się nie powiedzie.
        print(f"ERROR: Could not transcribe/analyze audio with Gemini: {e}")
        print(f"BŁĄD: Nie można przetranskrybować/analizować audio z Gemini: {e}")
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


def get_transcriber_function(transcriber_type="gemini_native"):
    # Check if the transcriber type is 'gemini_native'.
    # Sprawdza, czy typ transkrypcji to 'gemini_native'.
    if transcriber_type == "gemini_native":
        # Manually set the model to a more stable and proven one for audio.
        # Ręczne ustawienie modelu na bardziej stabilny i sprawdzony dla audio.
        supported_audio_model = "gemini-1.5-flash"

        print(f"Audio transcription model set to: {supported_audio_model}")
        print(f"Ustawiono model do transkrypcji audio na: {supported_audio_model}")

        # Define the inner function for Gemini native audio transcription.
        # Definiuje wewnętrzną funkcję do natywnej transkrypcji audio Gemini.
        def _transcribe_and_analyze_with_gemini_native(audio_file_path, log_data):
            # Upload the audio file to Gemini Files API.
            # Przesyła plik audio do Gemini Files API.
            file_obj = upload_audio_to_gemini_files_api(audio_file_path)
            if file_obj:
                log_data["gemini_file_name"] = file_obj.name
                try:
                    # Transcribe and analyze the audio using the Gemini model.
                    # Transkrybuje i analizuje audio za pomocą modelu Gemini.
                    analysis_text = transcribe_audio_with_gemini_model(file_obj, model_name=supported_audio_model,
                                                                       log_data=log_data)
                    return analysis_text
                finally:
                    # Ensure the file is deleted from Gemini Files API after processing.
                    # Upewnia się, że plik zostanie usunięty z Gemini Files API po przetworzeniu.
                    delete_gemini_file(file_obj)
            return None

        return _transcribe_and_analyze_with_gemini_native
    else:
        # Raise an error for an unknown transcriber type.
        # Podnosi błąd dla nieznanego typu transkrypcji.
        raise ValueError(f"Unknown transcription type: {transcriber_type}. Please choose 'gemini_native'.")
        raise ValueError(f"Nieznany typ transkrypcji: {transcriber_type}. Proszę wybrać 'gemini_native'.")


def process_audio_folder():
    print(f"\n--- Starting scan and processing of audio files from folder: {AUDIO_INPUT_FOLDER} ---")
    print(f"\n--- Rozpoczynam skanowanie i przetwarzanie plików audio z folderu: {AUDIO_INPUT_FOLDER} ---")

    # Supported audio file extensions.
    # Obsługiwane rozszerzenia plików audio.
    audio_files = [f for f in os.listdir(AUDIO_INPUT_FOLDER)
                   if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]

    if not audio_files:
        print("INFO: No audio files found in the folder for analysis.")
        print("INFO: Brak plików audio w folderze do analizy.")
        return

    # Get the transcriber function.
    # Pobiera funkcję do transkrypcji.
    transcribe_and_analyze_function = get_transcriber_function("gemini_native")

    total_files_processed = 0
    total_tokens_used = 0
    total_processing_time = 0.0

    # Global list to collect logs for all files.
    # Globalna lista do zbierania logów dla wszystkich plików.
    overall_logs = []

    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_INPUT_FOLDER, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_analysis_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.txt")
        log_file_path = os.path.join(AUDIO_LOG_FOLDER, f"{base_name}_log_{timestamp}.json")

        # Dictionary for logs of the current file.
        # Słownik na logi dla bieżącego pliku.
        current_log_data = {
            "timestamp": timestamp,
            "audio_file": audio_file,
            "status": "processing",
            "start_time_utc": datetime.datetime.utcnow().isoformat()
        }

        print(f"\n--- Processing file: {audio_file} ---")
        print(f"\n--- Przetwarzanie pliku: {audio_file} ---")
        file_start_time = time.time()

        analysis_result_text = None
        try:
            # Transcribe and analyze the audio file.
            # Transkrybuje i analizuje plik audio.
            analysis_result_text = transcribe_and_analyze_function(audio_path, current_log_data)

            if analysis_result_text:
                print(f"SUCCESS: Analysis of file '{audio_file}' completed.")
                print(f"SUKCES: Analiza pliku '{audio_file}' zakończona.")
                with open(output_analysis_path, "w", encoding="utf-8") as f:
                    f.write(analysis_result_text)
                print(f"Analysis result saved to: {output_analysis_path}")
                print(f"Wynik analizy zapisano do: {output_analysis_path}")
                current_log_data["status"] = "SUCCESS"
            else:
                print(f"WARNING: No analysis result for file '{audio_file}'.")
                print(f"OSTRZEŻENIE: Brak wyniku analizy dla pliku '{audio_file}'.")
                with open(output_analysis_path, "w", encoding="utf-8") as f:
                    f.write("[NO ANALYSIS RESULT FROM GEMINI]")
                    f.write("[BRAK WYNIKU ANALIZY Z GEMINI]")
                current_log_data["status"] = "WARNING_NO_RESULT"
        except Exception as e:
            print(f"ERROR: An error occurred while processing '{audio_file}': {e}")
            print(f"BŁĄD: Wystąpił błąd podczas przetwarzania '{audio_file}': {e}")
            with open(output_analysis_path, "w", encoding="utf-8") as f:
                f.write(f"[PROCESSING ERROR: {e}]")
                f.write(f"[BŁĄD PRZETWARZANIA: {e}]")
            current_log_data["status"] = "ERROR"
            current_log_data["exception_details"] = str(e)

        file_end_time = time.time()
        duration = file_end_time - file_start_time
        print(f"Processing time for '{audio_file}': {duration:.2f} seconds.")
        print(f"Czas przetwarzania dla '{audio_file}': {duration:.2f} sekundy.")

        current_log_data["duration_seconds"] = round(duration, 2)
        current_log_data["end_time_utc"] = datetime.datetime.utcnow().isoformat()

        # Save logs for the current file to a JSON file.
        # Zapisuje logi dla bieżącego pliku do pliku JSON.
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            json.dump(current_log_data, log_f, indent=4, ensure_ascii=False)
        print(f"Logs for file '{audio_file}' saved to: {log_file_path}")
        print(f"Logi dla pliku '{audio_file}' zapisano do: {log_file_path}")

        overall_logs.append(current_log_data)

        total_files_processed += 1
        total_processing_time += duration
        # Check if total_tokens is an integer before adding.
        # Sprawdza, czy total_tokens jest liczbą całkowitą przed dodaniem.
        if "total_tokens" in current_log_data and isinstance(current_log_data["total_tokens"], int):
            total_tokens_used += current_log_data["total_tokens"]

    print(f"\n--- Finished processing all audio files. ---")
    print(f"\n--- Zakończono przetwarzanie wszystkich plików audio. ---")
    print(f"Total files processed: {total_files_processed}")
    print(f"Łącznie przetworzono plików: {total_files_processed}")
    print(f"Total execution time: {total_processing_time:.2f} seconds.")
    print(f"Całkowity czas działania: {total_processing_time:.2f} sekundy.")
    print(f"Total token usage for all files: {total_tokens_used} tokens.")
    print(f"Całkowite zużycie tokenów dla wszystkich plików: {total_tokens_used} tokenów.")

    # Optionally: Save a summary log for the entire session.
    # Opcjonalnie: Zapisz sumaryczny log dla całej sesji.
    summary_log_path = os.path.join(AUDIO_LOG_FOLDER,
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
    # Call the main function to process the audio folder.
    # Wywołuje główną funkcję do przetwarzania folderu z audio.
    process_audio_folder()
    print("\n--- Script audio-text.py execution finished. ---")
    print("\n--- Zakończono działanie skryptu audio-text.py ---")