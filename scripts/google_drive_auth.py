from pydrive2.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # This will open a browser for authentication / To otworzy przeglądarkę do uwierzytelnienia
print("Authentication successful!") # Uwierzytelnienie zakończone sukcesem!