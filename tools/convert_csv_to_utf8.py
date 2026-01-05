from __future__ import annotations
import argparse
from pathlib import Path

def detect_encoding(b: bytes) -> str:
    # BOM quick checks
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return "utf-16"

    # Try strict decodes (avoid latin1 early because it almost always "works")
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp1252", "utf-16"]:
        try:
            b.decode(enc, errors="strict")
            return enc
        except UnicodeDecodeError:
            continue

    # Last resort fallback
    return "latin1"

def convert_to_utf8(path: Path, src_enc: str | None = None) -> None:
    raw = path.read_bytes()
    enc = src_enc or detect_encoding(raw)

    text = raw.decode(enc, errors="strict")
    # Write pure UTF-8 without BOM
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Converted {path} : {enc} -> utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to csv file")
    ap.add_argument("--src-enc", default=None, help="Force source encoding (e.g., gbk/cp1252)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(p)

    convert_to_utf8(p, args.src_enc)

if __name__ == "__main__":
    main()
