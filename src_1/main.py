import os, json
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from src_1.pdf_to_images import pdf_to_png_bytes  # ← ファイル名に合わせて修正
from src_1.classify_extract_vlm import run_vlm_variables, run_vlm_equations
from src_1.schema_docjson import VARIABLE_DB_SCHEMA, EQUATION_DB_SCHEMA

def validate_with(schema: dict, obj: dict) -> bool:
    Draft202012Validator(schema["schema"]).validate(obj)
    return True

if __name__ == "__main__":
    import argparse
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="input PDF")
    ap.add_argument("--dpi", type=int, default=220)

    # 出力先（既定ファイル名を要件に合わせる）
    ap.add_argument("--var-out", default="DB_variable.json", help="output JSON for variables")
    ap.add_argument("--eq-out",  default="DB_equation.json", help="output JSON for equations")
    args = ap.parse_args()

    pages = pdf_to_png_bytes(args.pdf, dpi=args.dpi)

    # 1) 変数DB
    var_db = run_vlm_variables(pages)
    validate_with(VARIABLE_DB_SCHEMA, var_db)
    with open(args.var_out, "w", encoding="utf-8") as f:
        json.dump(var_db, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.var_out}")

    # 2) 数式DB
    eq_db = run_vlm_equations(pages)
    validate_with(EQUATION_DB_SCHEMA, eq_db)
    with open(args.eq_out, "w", encoding="utf-8") as f:
        json.dump(eq_db, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.eq_out}")
