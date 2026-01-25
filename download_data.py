import os
import requests
import gzip
import shutil
import zipfile
from pathlib import Path

# Updated configuration using the reliable "Download All" ZIP links
DATASETS = {
    "BPI_2012": {
        "zip_url": "https://data.4tu.nl/ndownloader/items/533f66a4-8911-4ac7-8612-1235d65d1f37/versions/1",
        # The file inside the zip might handle spaces/underscores differently, so we'll search for it
        "final_filename": "BPI_Challenge_2012.xes"
    },
    "Sepsis": {
        "zip_url": "https://data.4tu.nl/ndownloader/items/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/versions/1",
        "final_filename": "Sepsis_Cases_-_Event_Log.xes"
    },
    "BPI_2018": {
        "zip_url": "https://data.4tu.nl/ndownloader/items/443451fd-d38a-4464-88b4-0fc641552632/versions/1",
        "final_filename": "BPI_Challenge_2018.xes"
    }
}

DOWNLOAD_DIR = Path("downloads")


def download_file(url, output_path):
    print(f"Downloading zip from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Saved zip to {output_path}")


def process_dataset(name, info):
    final_target = DOWNLOAD_DIR / info['final_filename']

    # Check if we already have the final .xes file
    if final_target.exists():
        print(f"[{name}] {final_target} already exists. Skipping.")
        return

    # 1. Download the Main ZIP
    temp_zip_path = DOWNLOAD_DIR / f"{name}_temp.zip"
    try:
        download_file(info['zip_url'], temp_zip_path)
    except Exception as e:
        print(f"[{name}] Failed to download: {e}")
        return

    # 2. Extract the .xes.gz file from the ZIP
    # We don't assume the exact internal filename; we look for the first .xes.gz we find.
    temp_gz_path = DOWNLOAD_DIR / f"{name}_temp.xes.gz"
    found_gz = False

    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zf:
            file_list = zf.namelist()
            # Find the file ending in .xes.gz (ignoring case)
            target_files = [f for f in file_list if f.lower().endswith('.xes.gz')]

            if not target_files:
                print(f"[{name}] Error: No .xes.gz file found inside the zip. Files found: {file_list}")
                return

            # Usually the largest one is the main log
            target_file = target_files[0]
            print(f"[{name}] Found log inside zip: {target_file}")

            with zf.open(target_file) as src, open(temp_gz_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            found_gz = True

    except Exception as e:
        print(f"[{name}] Error extracting zip: {e}")
        if temp_zip_path.exists(): os.remove(temp_zip_path)
        return

    # Remove the main zip to save space
    if temp_zip_path.exists():
        os.remove(temp_zip_path)

    if not found_gz:
        return

    # 3. Decompress the .gz to .xes
    print(f"[{name}] Decompressing GZIP to {final_target}...")
    try:
        with gzip.open(temp_gz_path, 'rb') as f_in:
            with open(final_target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"[{name}] Success!")
    except Exception as e:
        print(f"[{name}] Error decompressing GZIP: {e}")
    finally:
        # Cleanup the .gz file
        if temp_gz_path.exists():
            os.remove(temp_gz_path)


def main():
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    print(f"Download directory: {DOWNLOAD_DIR.resolve()}")

    for name, info in DATASETS.items():
        print(f"--- Processing {name} ---")
        process_dataset(name, info)


if __name__ == "__main__":
    main()