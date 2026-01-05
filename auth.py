import pickle
import os
import urllib.parse
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
alias = "eidosfinance"
token_file = f'token_{alias}.pickle'

def get_creds():
    flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)

    
    flow.redirect_uri = 'http://127.0.0.1:8080'

    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')

    print("-" * 60)
    print("LANGKAH LOGIN FINAL:")
    print(f"1. Buka link ini di browser laptop lu:\n\n{auth_url}\n")
    print("2. Login dan klik 'Allow'.")
    print("3. Browser lu akan ERROR (Site can't be reached) di alamat http://127.0.0.1:8080/...")
    print("4. COPY SELURUH URL yang ada di Address Bar browser lu.")
    print("-" * 60)

    full_url = input("PASTE SELURUH URL DARI ADDRESS BAR DI SINI: ").strip()

    try:
        if "code=" in full_url:
            code = full_url.split("code=")[1].split("&")[0]
            code = urllib.parse.unquote(code)
        else:
            code = full_url

        flow.fetch_token(code=code)
        creds = flow.credentials

        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
        print(f"\n[ BERHASIL ] File {token_file} sudah dibuat di server.")
    except Exception as e:
        print(f"\n[ ERROR ] Gagal: {e}")

if __name__ == "__main__":
    get_creds()
