import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# --- 設定 ---
DATA_ROOT = Path("./data/nist")
EXTRACT_ROOT = DATA_ROOT / "extracted"
BY_CLASS_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"[Skip] {dest_path.name} already exists.")
        return

    print(f"[Download] Downloading {url} ...")
    # 中断時に不完全なファイルが残って次回スキップされないよう、一時ファイルに書き込む
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                tmp_path.open("wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, ncols=80) as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp_path.rename(dest_path)
        print("Done.")
    except (requests.RequestException, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        print(f"[Error] Failed to download {url}: {e}")


def extract_file(zip_path: Path, extract_to: Path):
    print(f"[Extract] Extracting {zip_path.name} into {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as z:
            for member in tqdm(z.infolist(), desc="Files", ncols=80):
                z.extract(member, extract_to)
        print("Done.")
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        print(f"[Error] Failed to extract {zip_path}: {e}")


def main():
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    zip_class = DATA_ROOT / "by_class.zip"
    download_file(BY_CLASS_URL, zip_class)
    extract_file(zip_class, EXTRACT_ROOT)

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print(f"Files extracted into: {EXTRACT_ROOT}")
    print("Structure should be:")
    print(f"  {EXTRACT_ROOT / 'by_class'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
