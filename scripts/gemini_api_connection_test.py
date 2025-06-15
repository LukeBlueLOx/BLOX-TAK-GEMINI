import uno
import unohelper
import json
import http.client  # For making HTTP requests
import yaml

with open("config.yaml", "r") as cr:
    config_vals = yaml.full_load(cr)
KEY = config_vals['KEY']

# Main macro function to test Google Gemini API connection
# Główna funkcja makra do testowania połączenia z API Google Gemini
def test_gemini_connection(*args):
    # --- GEMINI API KEY CONFIGURATION ---
    # --- KONFIGURACJA KLUCZA API GEMINI ---
    # IMPORTANT: Replace "*****" with your real API key in config.yaml!
    # WAŻNE: Zastąp "*****" swoim prawdziwym kluczem API w pliku config.yaml!
    GOOGLE_API_KEY = KEY
    print("Gemini API configured successfully.")
    print("Gemini API skonfigurowane pomyślnie.")

    API_HOST = "generativelanguage.googleapis.com"
    MODEL_NAME = "gemini-1.5-flash-8b"  # Using a simple model for testing
    # Używamy prostego modelu do testu

    # A very simple prompt to test the connection
    # Bardzo prosty prompt do przetestowania połączenia
    test_prompt = "Say 'OK'." # English version for the model
    test_prompt_pl = "Powiedz 'OK'." # Polish version for clarity in context

    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": test_prompt_pl} # Use the Polish prompt for the model
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,  # Low temperature for predictable output
            # Niska temperatura dla przewidywalnego wyniku
            "maxOutputTokens": 50,  # Short response
            # Krótka odpowiedź
        }
    }

    try:
        conn = http.client.HTTPSConnection(API_HOST)
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GOOGLE_API_KEY
        }

        endpoint = f"/v1beta/models/{MODEL_NAME}:generateContent"
        body = json.dumps(request_body)

        print(f"Testing connection to Google Gemini API at {API_HOST}{endpoint}...")
        print(f"Testuję połączenie z Google Gemini API na {API_HOST}{endpoint}...")

        conn.request("POST", endpoint, body=body, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode('utf-8')
        conn.close()

        print(f"Gemini API Response Status: {response.status}")
        print(f"Status odpowiedzi API Gemini: {response.status}")
        print(f"Full Gemini API Response (for diagnostics): {response_data}")
        print(f"Pełna odpowiedź API Gemini (do diagnostyki): {response_data}")

        if response.status == 200:
            result = json.loads(response_data)
            # Check if the response contains the expected text
            # Sprawdzamy, czy odpowiedź zawiera oczekiwany tekst
            if 'candidates' in result and len(result['candidates']) > 0 \
                    and 'content' in result['candidates'][0] \
                    and 'parts' in result['candidates'][0]['content'] \
                    and len(result['candidates'][0]['content']['parts']) > 0 \
                    and 'text' in result['candidates'][0]['content']['parts'][0]:

                model_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                if "OK" in model_response.upper():  # Check if the model returned "OK"
                    # Sprawdzamy, czy model zwrócił "OK"
                    print("SUCCESS: Connection to Google Gemini API successful! Model responded: " + model_response)
                    print("SUKCES: Połączenie z Google Gemini API zakończone sukcesem! Model odpowiedział: " + model_response)
                else:
                    print(f"WARNING: Google Gemini API connection successful, but model did not return 'OK'. Response: {model_response}")
                    print(f"OSTRZEŻENIE: Połączenie z Google Gemini API zakończone sukcesem, ale model nie zwrócił 'OK'. Odpowiedź: {model_response}")
            else:
                print(f"ERROR: Successful connection, but invalid Gemini response structure. Status: {response.status}")
                print(f"BŁĄD: Pomyślne połączenie, ale nieprawidłowa struktura odpowiedzi Gemini. Status: {response.status}")

        elif response.status == 400:  # Bad Request (e.g., wrong API key, bad request format)
            # Bad Request (np. błędny klucz API, zły format zapytania)
            error_details = json.loads(response_data).get('error', {}).get('message', 'No details available.')
            # Brak szczegółów.
            print(f"ERROR: Gemini API Error (400 - Bad Request). Check API key and request format. Details: {error_details}")
            print(f"BŁĄD: Błąd API Gemini (400 - Bad Request). Sprawdź klucz API i format zapytania. Szczegóły: {error_details}")
        elif response.status == 403:  # Forbidden (e.g., API key lacks permissions or API is not enabled)
            # Forbidden (np. klucz API nie ma uprawnień lub API nie jest włączone)
            print(f"ERROR: Gemini API Error (403 - Forbidden). API key might lack permissions or API is not enabled. Response: {response_data}")
            print(f"BŁĄD: Błąd API Gemini (403 - Forbidden). Klucz API może nie mieć uprawnień lub API nie jest włączone. Odpowiedź: {response_data}")
        elif response.status == 404:  # Not Found (e.g., wrong URL, model does not exist)
            # Not Found (np. zły URL, model nie istnieje)
            print(f"ERROR: Gemini API Error (404 - Not Found). Check model name ({MODEL_NAME}) or API address. Response: {response_data}")
            print(f"BŁĄD: Błąd API Gemini (404 - Not Found). Sprawdź nazwę modelu ({MODEL_NAME}) lub adres API. Odpowiedź: {response_data}")
        else:
            print(f"ERROR: Google Gemini server error: HTTP {response.status} - {response_data}")
            print(f"BŁĄD: Błąd z serwera Google Gemini: HTTP {response.status} - {response_data}")

    except Exception as e:
        print(f"CRITICAL ERROR: During communication with Google Gemini: {e}")
        print(f"KRYTYCZNY BŁĄD: Podczas komunikacji z Google Gemini: {e}")


# Register the macro for LibreOffice
# Rejestracja makra dla LibreOffice
g_exportedScripts = test_gemini_connection,