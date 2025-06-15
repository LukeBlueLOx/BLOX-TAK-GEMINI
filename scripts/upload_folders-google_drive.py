import os
import yaml
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def upload_folder_to_drive(drive, local_path, parent_folder_id, folder_name):
    """
    # Uploads a local folder and its contents to Google Drive.
    # Przesyła lokalny folder i jego zawartość na Google Drive.
    """
    # Create the folder on Google Drive
    # Utwórz folder na Google Drive
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_folder_id}]
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    print(f"Created folder on Google Drive: '{folder_name}' (ID: {folder['id']})")
    print(f"Utworzono folder na Google Drive: '{folder_name}' (ID: {folder['id']})")

    # Upload files and subfolders
    # Prześlij pliki i podfoldery
    for item_name in os.listdir(local_path):
        item_path = os.path.join(local_path, item_name)
        if os.path.isfile(item_path):
            # Upload file
            # Prześlij plik
            file_metadata = {
                'title': item_name,
                'parents': [{'id': folder['id']}]
            }
            gfile = drive.CreateFile(file_metadata)
            gfile.SetContentFile(item_path)
            gfile.Upload()
            print(f"  Uploaded file: '{item_name}'")
            print(f"  Przesłano plik: '{item_name}'")
        elif os.path.isdir(item_path):
            # Recursively upload subfolder
            # Rekurencyjnie prześlij podfolder
            print(f"  Uploading subfolder: '{item_name}'...")
            print(f"  Przesyłam podfolder: '{item_name}'...")
            upload_folder_to_drive(drive, item_path, folder['id'], item_name)


def main():
    # Load configuration
    # Wczytaj konfigurację
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: 'config.yaml' file not found. Make sure it exists in the same directory as the script.")
        print("Błąd: Plik 'config.yaml' nie znaleziony. Upewnij się, że istnieje w tym samym katalogu co skrypt.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing 'config.yaml' file: {e}")
        print(f"Błąd podczas parsowania pliku 'config.yaml': {e}")
        return

    folders_to_check = config.get('folders_to_check', [])
    base_path = config.get('base_path', os.getcwd())
    google_drive_parent_folder_id = config.get('google_drive_parent_folder_id')

    if not google_drive_parent_folder_id:
        print("Error: 'google_drive_parent_folder_id' not found in config.yaml.")
        print("Błąd: 'google_drive_parent_folder_id' nie został znaleziony w pliku config.yaml.")
        return

    # Authenticate with Google Drive
    # Uwierzytelnij się z Google Drive
    gauth = GoogleAuth()
    # Try to load saved credentials
    # Spróbuj załadować zapisane poświadczenia
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        # Authenticate if no credentials found
        # Uwierzytelnij, jeśli nie znaleziono poświadczeń
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh token if expired
        # Odśwież token, jeśli wygasł
        gauth.Refresh()
    else:
        # Initialize the saved creds
        # Zainicjuj zapisane poświadczenia
        gauth.Authorize()
    gauth.SaveCredentialsFile(
        "credentials.json")  # Save credentials for next run / Zapisz poświadczenia do następnego uruchomienia

    drive = GoogleDrive(gauth)

    # Get current date for the target folder name
    # Pobierz aktualną datę dla nazwy folderu docelowego
    current_date_folder_name = datetime.now().strftime("%Y-%m-%d")

    # Create the main date folder on Google Drive
    # Utwórz główny folder z datą na Google Drive
    date_folder_metadata = {
        'title': current_date_folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': google_drive_parent_folder_id}]
    }
    date_folder = drive.CreateFile(date_folder_metadata)
    date_folder.Upload()
    print(f"\nMain folder created on Google Drive: '{current_date_folder_name}' (ID: {date_folder['id']})")
    print(f"\nUtworzono główny folder na Google Drive: '{current_date_folder_name}' (ID: {date_folder['id']})")

    # Check and upload folders
    # Sprawdź i prześlij foldery
    for folder_name in folders_to_check:
        full_path = os.path.join(base_path, folder_name)
        print(f"\nChecking folder: '{full_path}'")
        print(f"\nSprawdzam folder: '{full_path}'")

        if not os.path.exists(full_path):
            print(f"  Folder does not exist. Skipping.")
            print(f"  Folder nie istnieje. Pomijam.")
            continue

        if not os.path.isdir(full_path):
            print(f"  This is not a directory. Skipping.")
            print(f"  To nie jest katalog. Pomijam.")
            continue

        if not os.listdir(full_path):
            print(f"  Folder is empty. Skipping.")
            print(f"  Folder jest pusty. Pomijam.")
        else:
            print(f"  Folder is not empty. Starting upload...")
            print(f"  Folder nie jest pusty. Rozpoczynam przesyłanie...")
            try:
                upload_folder_to_drive(drive, full_path, date_folder['id'], folder_name)
                print(f"  Finished uploading folder '{folder_name}'.")
                print(f"  Zakończono przesyłanie folderu '{folder_name}'.")
            except Exception as e:
                print(f"  An error occurred while uploading folder '{folder_name}': {e}")
                print(f"  Wystąpił błąd podczas przesyłania folderu '{folder_name}': {e}")

    print("\nProcess finished.")
    print("\nProces zakończony.")


if __name__ == '__main__':
    main()