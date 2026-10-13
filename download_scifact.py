import os
import urllib.request
import zipfile

DATA_DIR = "data/raw/scifact"

os.makedirs(DATA_DIR, exist_ok=True)

url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
output_file = "data/raw/scifact/data.tar.gz"

print("📥 Downloading SciFact dataset...")

urllib.request.urlretrieve(url, output_file)

print("✅ Download completed!")
print("📦 Extracting dataset...")

import tarfile

with tarfile.open(output_file, "r:gz") as tar:
    tar.extractall(DATA_DIR)

print("🎉 SciFact dataset extracted successfully!")

print("\nFiles found:")

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        print(os.path.join(root, file))