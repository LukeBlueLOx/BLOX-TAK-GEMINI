import uno
import unohelper
import json
import http.client
import yaml

with open("config.yaml", "r") as cr:
    config_vals = yaml.full_load(cr)
KEY = config_vals['KEY']

# Main macro function to correct the entire document text using Google Gemini
# Główna funkcja makra do poprawiania całego tekstu dokumentu za pomocą Google Gemini
def correct_entire_document_text_google(*args):
    # Get the current document
    # Pobierz bieżący dokument
    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()
    if not model:
        print("ERROR: No document open.")
        print("BŁĄD: Brak otwartego dokumentu.")
        return

    # Get the entire text object of the document
    # Pobierz obiekt całego tekstu dokumentu
    text_object = model.Text
    if not text_object:
        print("ERROR: Could not get the document's text object.")
        print("BŁĄD: Nie można uzyskać obiektu tekstu dokumentu.")
        return

    # Get the full text content from the document
    # Pobierz pełną zawartość tekstową z dokumentu
    full_document_text = text_object.getString()

    if not full_document_text.strip():
        print("INFO: The current document is empty or contains only whitespace. Nothing to correct.")
        print("INFO: Bieżący dokument jest pusty lub zawiera tylko białe znaki. Brak tekstu do poprawy.")
        return

    print(f"Text to correct (first 100 characters): {full_document_text[:100]}...")
    print(f"Tekst do poprawy (pierwsze 100 znaków): {full_document_text[:100]}...")

    # --- GEMINI API KEY CONFIGURATION ---
    # --- KONFIGURACJA KLUCZA API GEMINI ---
    # IMPORTANT: Replace "*****" with your real API key in config.yaml!
    # WAŻNE: Zastąp "*****" swoim prawdziwym kluczem API w pliku config.yaml!
    GOOGLE_API_KEY = KEY
    print("Gemini API configured successfully.")
    print("Gemini API skonfigurowane pomyślnie.")

    API_HOST = "generativelanguage.googleapis.com"
    MODEL_NAME = "gemini-1.5-flash-8b"

    # Prepare the prompt for Gemini
    # Przygotuj prompt dla Gemini
    prompt_content = (
        f"Correct spelling, grammar, and stylistic errors in the following Polish text. "
        f"Ensure the text is grammatically correct and sounds natural in Polish. "
        f"Return only the corrected text, without additional comments. Text to correct:\n\n{full_document_text}"
    )
    prompt_content_pl = (
        f"Popraw błędy ortograficzne, gramatyczne i stylistyczne w poniższym tekście. "
        f"Upewnij się, że tekst jest poprawny językowo i naturalnie brzmiący po polsku. "
        f"Zwróć tylko poprawiony tekst, bez dodatkowych komentarzy. Tekst do poprawy:\n\n{full_document_text}"
    )

    # Structure of the request body for Gemini API
    # Struktura ciała żądania dla API Gemini
    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_content_pl}  # Using the Polish prompt version
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        }
    }

    try:
        # Connect to Gemini API
        # Połącz się z API Gemini
        conn = http.client.HTTPSConnection(API_HOST)
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GOOGLE_API_KEY
        }
        endpoint = f"/v1beta/models/{MODEL_NAME}:generateContent"
        body = json.dumps(request_body)

        print(f"Sending request to {API_HOST}{endpoint}...")
        print(f"Wysyłam zapytanie do {API_HOST}{endpoint}...")

        conn.request("POST", endpoint, body=body, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode('utf-8')
        conn.close()

        print(f"Gemini API Response Status: {response.status}")
        print(f"Status odpowiedzi API Gemini: {response.status}")
        print(f"Full Gemini API Response (for diagnostics): {response_data}")
        print(f"Pełna odpowiedź API Gemini (do diagnostyki): {response_data}")

        # Check if the response is successful
        # Sprawdź, czy odpowiedź jest pomyślna
        if response.status == 200:
            result = json.loads(response_data)
            corrected_text = ""

            # Extract the corrected text from the response
            # Wyodrębnij poprawiony tekst z odpowiedzi
            if 'candidates' in result and len(result['candidates']) > 0 \
                    and 'content' in result['candidates'][0] \
                    and 'parts' in result['candidates'][0]['content'] \
                    and len(result['candidates'][0]['content']['parts']) > 0 \
                    and 'text' in result['candidates'][0]['content']['parts'][0]:

                corrected_text = result['candidates'][0]['content']['parts'][0]['text'].strip()

                if corrected_text:
                    # Replace the entire document text with the corrected text
                    # Zastąp cały tekst dokumentu poprawionym tekstem
                    text_object.setString(corrected_text)
                    print("SUCCESS: Entire document text corrected successfully by Google Gemini!")
                    print("SUKCES: Cały tekst dokumentu poprawiony pomyślnie przez Google Gemini!")
                else:
                    print("WARNING: Gemini returned an empty corrected text.")
                    print("OSTRZEŻENIE: Gemini zwróciło pusty poprawiony tekst.")
            else:
                print("ERROR: Invalid Gemini API response structure (missing 'candidates' or 'content').")
                print("BŁĄD: Nieprawidłowa struktura odpowiedzi API Gemini (brak 'candidates' lub 'content').")

            # Display token usage
            # Wyświetl zużycie tokenów
            if 'usageMetadata' in result:
                usage = result['usageMetadata']
                usage_msg = (
                    f"Model: {result.get('modelVersion', 'Unknown version')}\n"
                    f"Input Tokens: {usage.get('promptTokenCount', 0)}\n"
                    f"Output Tokens: {usage.get('candidatesTokenCount', 0)}\n"
                    f"Total Tokens: {usage.get('totalTokenCount', 0)}"
                )
                print(f"--- Gemini Resource Usage ---\n{usage_msg}")
                print(f"--- Zużycie zasobów Gemini ---\n{usage_msg}")
            else:
                print("INFO: No token usage information in Gemini response.")
                print("INFO: Brak informacji o zużyciu tokenów w odpowiedzi Gemini.")

        else:
            print(f"ERROR: Google Gemini server error: HTTP {response.status} - {response_data}")
            print(f"BŁĄD: Błąd serwera Google Gemini: HTTP {response.status} - {response_data}")

    except Exception as e:
        print(f"CRITICAL ERROR: During communication with Google Gemini: {e}")
        print(f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini: {e}")


# Register the macro for LibreOffice
# Rejestracja makra dla LibreOffice
g_exportedScripts = correct_entire_document_text_google,