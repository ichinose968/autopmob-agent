import os, base64, json, uuid
from typing import List, Tuple
from openai import OpenAI
from src.schema_docjson import DOCJSON_SCHEMA

MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

def _b64png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def build_input(messages: List[dict]) -> list:
    # Responses API の input 形式
    return messages

SYSTEM_INSTRUCTIONS = (
    "You are a scientific VLM preprocessor. "
    "From page images of a technical paper, produce a JSON matching the schema. "
    "Keep equations as LaTeX when you can. Extract sections with clean text, "
    "and include best-effort quantities (value+unit mentions) and minimal provenance. "
    "Language normalization should be 'en' if the paper is not English."
)

# def run_vlm_on_pages(pages: List[Tuple[int, bytes]]) -> dict:
#     client = OpenAI()
#     content = [{"type": "input_text", "text": "Extract DocJSON from these pages (first pass)."}]
#     for idx, png in pages:
#         content.append({
#             "type": "input_image",
#             "image_url": _b64png(png),
#             # 必要に応じて detail を調整（仕様はモデルにより異なるため最新ドキュメント参照）
#             # "detail": "auto"
#         })



#         resp = client.responses.create(
#         model=MODEL,
#         input=[
#             {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS}]},
#             {"role": "user", "content": content}
#         ],
#         response_format={
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "docjson",
#                 "schema": DOCJSON_SCHEMA
#             }
#         }
#     )
#     data = resp.output_text  # ここが dict の場合もあるので必要なら分岐


#     # SDKは output_text でJSON文字列を返す実装（バージョンにより異なる）—最新リファレンス確認
#     doc = json.loads(data) if isinstance(data, str) else data
#     # doc_id 付与と簡易プロヴナンス
#     doc.setdefault("doc_id", str(uuid.uuid4()))
#     prov = doc.get("provenance", [])
#     prov.append({"note": "generated_by_vlm", "pages": [p for p, _ in pages]})
#     doc["provenance"] = prov
#     return doc

def run_vlm_on_pages(pages):
    client = OpenAI()

    # ユーザーメッセージの content（テキスト + 画像たち）
    content = [{"type": "input_text", "text": "Extract DocJSON from these pages (first pass)."}]
    for idx, png in pages:
        content.append({
            "type": "input_image",
            "image_url": _b64png(png),  # "data:image/png;base64,...."
            # "detail": "auto",  # 必要なら
        })

    # ★ Responses API + tools(function) で DocJSON を“引数”として返させる
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS}]},
            {"role": "user",   "content": content},
        ],
        tools=[{
            "type": "function",
            "name": "emit_docjson",
            "description": "Return the extracted document strictly as DocJSON.",
            # ★ ラッパーを剥がして“スキーマ本体”を渡す
            "parameters": DOCJSON_SCHEMA["schema"]   # ← ここがポイント
        }],
        tool_choice="required"
    )


    # ---- ツール引数（= DocJSON本体）を取り出すユーティリティ ----
    def _extract_tool_args(r):
        """
        SDKバージョン差異に耐えるため、いくつかの取り出し経路を順に試す。
        成功したら Python dict を返す。
        """
        # 0) トップレベル output[*] に function_call が直接入っているパターン（今回これ）
        try:
            for item in (getattr(r, "output", None) or []):
                # SDKによって dict/obj の差があり得るので両対応
                item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                if item_type in ("function_call", "tool_call"):  # 念のため両方
                    args = getattr(item, "arguments", None) or (isinstance(item, dict) and item.get("arguments"))
                    if args:
                        return json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass

        # 1) 標準的: output -> content -> tool_calls
        try:
            tool_calls = r.output[0].content[0].tool_calls
            if tool_calls:
                args = tool_calls[0].function.arguments
                return json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass

        # 2) output 直下に tool_calls
        try:
            tool_calls = r.output[0].tool_calls
            if tool_calls:
                args = tool_calls[0].function.arguments
                return json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass

        # 3) content を総なめして tool_calls を探す
        try:
            for block in r.output[0].content:
                if hasattr(block, "tool_calls") and block.tool_calls:
                    args = block.tool_calls[0].function.arguments
                    return json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass

        raise RuntimeError("Could not extract tool/function arguments from Responses API result.")
    

    doc = _extract_tool_args(resp)

    # 追記（任意）
    doc.setdefault("doc_id", str(uuid.uuid4()))
    prov = doc.get("provenance", [])
    prov.append({"note": "generated_by_vlm", "pages": [p for p, _ in pages]})
    doc["provenance"] = prov

    return doc