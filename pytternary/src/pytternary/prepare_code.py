import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data"

TEXT_PATH = DATA / "code_dataset.txt"
TOKENIZER_PATH = DATA / "tokenizer_code.json"
ENCODED_PATH = DATA / "code_encoded_u16.bin"
VOCAB_SIZE = 4096

BASE = "https://huggingface.co/datasets/codeparrot/xlcost-text-to-code/resolve/main/data"

SUBSETS = [
    "C++-snippet-level", "C++-program-level",
    "C-snippet-level", "C-program-level",
    "Csharp-snippet-level", "Csharp-program-level",
    "Java-snippet-level", "Java-program-level",
    "Javascript-snippet-level", "Javascript-program-level",
    "PHP-snippet-level", "PHP-program-level",
    "Python-snippet-level", "Python-program-level",
]


def download_jsonl(url: str) -> list[dict]:
    print(f"    Downloading {url.split('/')[-1]}...")
    resp = urllib.request.urlopen(url)
    text = resp.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def extract_code():
    if TEXT_PATH.exists():
        print(f"Text file already exists: {TEXT_PATH}")
        return

    total = 0
    with open(TEXT_PATH, "w") as f:
        for subset in SUBSETS:
            for split in ("train", "test", "valid"):
                url = f"{BASE}/{subset}/{split}.json"
                try:
                    records = download_jsonl(url)
                except Exception as e:
                    print(f"    Skipping {url}: {e}")
                    continue
                for rec in records:
                    code = rec.get("code")
                    if code and code != "null":
                        cleaned = (
                            code.replace("NEW_LINE", "\n")
                            .replace("INDENT", "    ")
                            .replace("DEDENT", "")
                            .replace("  \n", "\n")
                        )
                        f.write(cleaned)
                        f.write("\n\n")
                        total += 1
                print(f"      {len(records)} rows")
    print(f"Extracted {total} code snippets to {TEXT_PATH}")


def train_tokenizer():
    if TOKENIZER_PATH.exists():
        print(f"Tokenizer already exists: {TOKENIZER_PATH}")
        return
    from pytternary_tokenizer import train_tokenizer as train_rs
    print("Training BPE tokenizer...")
    train_rs(str(TEXT_PATH), str(TOKENIZER_PATH), VOCAB_SIZE)
    print(f"  Saved to {TOKENIZER_PATH}")


def encode_dataset():
    if ENCODED_PATH.exists():
        print(f"Encoded dataset already exists: {ENCODED_PATH}")
        return
    from pytternary_tokenizer import tokenize_dataset_file
    print("Encoding dataset (parallel)...")
    n = tokenize_dataset_file(str(TEXT_PATH), str(TOKENIZER_PATH), str(ENCODED_PATH), None)
    print(f"  Encoded {n} tokens to {ENCODED_PATH}")


def main():
    print("=== Preparing code dataset ===\n")
    print("Downloading xlcost JSON files...")
    extract_code()
    print()
    train_tokenizer()
    print()
    encode_dataset()
    print("\nDone. Train with:\n"
          "  uv run python -m pytternary.train --dataset code --steps 500")


if __name__ == "__main__":
    main()
