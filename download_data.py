import requests
import zipfile
import io
import os

url = "https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip"
print(f"Downloading from {url}...")
r = requests.get(url)
print("Download complete. Extracting...")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(".")
print("Extraction complete.")
