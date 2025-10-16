import os, base64, json, uuid
from typing import List, Tuple
from openai import OpenAI
from src_1.schema_docjson import VARIABLE_DB_SCHEMA, EQUATION_DB_SCHEMA

MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

def _b64png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def _pages_to_user_content(pages):
    content = [{"type": "input_text", "text": "Extract only what is asked."}]
    for idx, png in pages:
        content.append({"type": "input_image", "image_url": _b64png(png)})
    return content

_SYSTEM_VARS = (
    "You are a scientific VLM. "
    "TASK: From page images of a technical/scientific paper, extract ONLY the VARIABLES used in the paper. "
    "OUTPUT: JSON strictly matching the given variable_db schema. "
    "For each variable, provide: name (canonical name in English if possible, otherwise original), symbol (as appears; LaTeX allowed), "
    "and a concise description (1-3 sentences). Include unit if identifiable. Ignore all narrative text unrelated to variables. "
    "Do not include equations or unrelated sections. Keep it minimal and consistent."
)

_SYSTEM_EQS = (
    "You are a scientific VLM. "
    "TASK: From page images of a technical/scientific paper, extract ONLY the EQUATIONS. "
    "OUTPUT: JSON strictly matching the given equation_db schema. "
    "For each equation, return its LaTeX (no numbering artifacts) and a concise description (1-3 sentences) explaining its role/meaning. "
    "Optionally include variables (symbols appearing in the equation) and page index if clear. "
    "Exclude narrative paragraphs and any non-equation content."
)

def _extract_tool_args(r):
    # 既存の堅牢抽出。SDK差分に耐える。
    try:
        for item in (getattr(r, "output", None) or []):
            item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if item_type in ("function_call", "tool_call"):
                args = getattr(item, "arguments", None) or (isinstance(item, dict) and item.get("arguments"))
                if args:
                    return json.loads(args) if isinstance(args, str) else args
    except Exception:
        pass
    try:
        tool_calls = r.output[0].content[0].tool_calls
        if tool_calls:
            args = tool_calls[0].function.arguments
            return json.loads(args) if isinstance(args, str) else args
    except Exception:
        pass
    try:
        tool_calls = r.output[0].tool_calls
        if tool_calls:
            args = tool_calls[0].function.arguments
            return json.loads(args) if isinstance(args, str) else args
    except Exception:
        pass
    try:
        for block in r.output[0].content:
            if hasattr(block, "tool_calls") and block.tool_calls:
                args = block.tool_calls[0].function.arguments
                return json.loads(args) if isinstance(args, str) else args
    except Exception:
        pass
    raise RuntimeError("Could not extract tool/function arguments from Responses API result.")

def run_vlm_variables(pages: List[Tuple[int, bytes]]) -> dict:
    client = OpenAI()
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": _SYSTEM_VARS}]},
            {"role": "user",   "content": _pages_to_user_content(pages)},
        ],
        tools=[{
            "type": "function",
            "name": "emit_variable_db",
            "description": "Return the variable database strictly as variable_db schema.",
            "parameters": VARIABLE_DB_SCHEMA["schema"]
        }],
        tool_choice="required"
    )
    doc = _extract_tool_args(resp)
    doc.setdefault("doc_id", str(uuid.uuid4()))
    prov = doc.get("provenance", [])
    prov.append({"note": "generated_by_vlm_variables", "pages": [p for p, _ in pages]})
    doc["provenance"] = prov
    return doc

def run_vlm_equations(pages: List[Tuple[int, bytes]]) -> dict:
    client = OpenAI()
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": _SYSTEM_EQS}]},
            {"role": "user",   "content": _pages_to_user_content(pages)},
        ],
        tools=[{
            "type": "function",
            "name": "emit_equation_db",
            "description": "Return the equation database strictly as equation_db schema.",
            "parameters": EQUATION_DB_SCHEMA["schema"]
        }],
        tool_choice="required"
    )
    doc = _extract_tool_args(resp)
    doc.setdefault("doc_id", str(uuid.uuid4()))
    prov = doc.get("provenance", [])
    prov.append({"note": "generated_by_vlm_equations", "pages": [p for p, _ in pages]})
    doc["provenance"] = prov
    return doc
