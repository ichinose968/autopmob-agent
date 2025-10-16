import os, json
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from src.pdf_to_images import pdf_to_png_bytes
from src.classify_extract_vlm import run_vlm_on_pages
from src.schema_docjson import DOCJSON_SCHEMA

def validate(doc: dict):
    Draft202012Validator(DOCJSON_SCHEMA["schema"]).validate(doc)
    return True

if __name__ == "__main__":
    import argparse
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="input PDF")
    ap.add_argument("-o", "--out", default="doc.json")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    pages = pdf_to_png_bytes(args.pdf, dpi=args.dpi)
    doc = run_vlm_on_pages(pages)
    validate(doc)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

