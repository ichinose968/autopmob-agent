# pip install langchain langgraph pydantic pydantic-core
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

# Ensure project root (parent of this file's directory) is importable when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =====================
# 1) 共有状態の定義
# =====================
class EquationItem(BaseModel):
    id: str
    latex: str
    context: str
    page: Optional[int] = None
    variables: List[str] = Field(default_factory=list)

class CanonicalEquation(BaseModel):
    key: str                   # 正規化キー（例：構造木/トークン列ハッシュ）
    latex: str                 # 正規化済みLaTeX
    support_ids: List[str]     # 統合元 EquationItem.id
    role: Optional[str] = None # 例：mass_balance, rate_law など

class CompiledModel(BaseModel):
    title: str
    entities: Dict[str, Any]   # 例：{"variables": [...], "parameters": [...]}
    equations: List[Dict[str, Any]]  # 例：{"latex": "...", "type": "...", "dependencies": [...]}
    notes: Optional[str] = None

class PipelineState(TypedDict):
    doc_id: str
    source_meta: Dict[str, Any]
    passages: List[Dict[str, Any]]     # [{"text": "...", "page": 3}, ...]
    raw_equations: List[Dict[str, Any]]
    canonical_equations: List[Dict[str, Any]]
    compiled_model: Dict[str, Any]
    logs: List[str]

# =====================
# 2) 既存関数を差し込む“受け口”
# =====================
def run_extractor(passages: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[EquationItem]:
    """
    passages から LaTeX らしき表現を拾い、EquationItem のリストとして返す簡易実装。
    実運用では既存の VLM パイプライン等に差し替えてください。
    """
    latex_pattern = re.compile(r"\${1,2}([^$]+)\${1,2}")
    var_pattern = re.compile(r"[A-Za-z]+(?:_[A-Za-z0-9]+)?")

    doc_hint = meta.get("doc_id") or meta.get("title") or "doc"
    items: List[EquationItem] = []

    for passage_idx, passage in enumerate(passages):
        text = passage.get("text", "") or ""
        page = passage.get("page")
        matches = latex_pattern.findall(text)

        # 数式らしきパターンが見つからない場合は文章全体を候補として扱う
        if not matches and text.strip():
            matches = [text.strip()]

        for eq_idx, expr in enumerate(matches):
            expr = expr.strip()
            if not expr:
                continue

            variables = sorted(set(var_pattern.findall(expr)))
            eq_id = f"{doc_hint}-p{passage_idx}-e{eq_idx}"
            items.append(
                EquationItem(
                    id=eq_id,
                    latex=expr,
                    context=text,
                    page=page,
                    variables=variables,
                )
            )

    return items


def write_extraction_summary(
    summary_path: Path,
    var_db: Dict[str, Any],
    eq_db: Dict[str, Any],
    var_out: str,
    eq_out: str,
) -> None:
    """Write a compact JSON summary describing the extracted artifacts."""
    summary = {
        "variables": {
            "output_file": var_out,
            "count": len(var_db.get("variables", [])),
            "preview": var_db.get("variables", [])[:5],
        },
        "equations": {
            "output_file": eq_out,
            "count": len(eq_db.get("equations", [])),
            "preview": eq_db.get("equations", [])[:5],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def extractor_cli():
    """CLI から既存 VLM ベースの抽出処理を呼び出すためのラッパー。"""
    from dotenv import load_dotenv
    from jsonschema import Draft202012Validator
    from src_1.pdf_to_images import pdf_to_png_bytes  # ← ファイル名に合わせて修正
    from src_1.classify_extract_vlm import run_vlm_variables, run_vlm_equations
    from src_1.schema_docjson import VARIABLE_DB_SCHEMA, EQUATION_DB_SCHEMA

    def validate_with(schema: dict, obj: dict) -> bool:
        Draft202012Validator(schema["schema"]).validate(obj)
        return True

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="input PDF")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--var-out", default="DB_variable.json", help="output JSON for variables")
    parser.add_argument("--eq-out", default="DB_equation.json", help="output JSON for equations")
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="optional path to a JSON summary describing the extracted variables/equations",
    )
    args = parser.parse_args()

    load_dotenv()
    pages = pdf_to_png_bytes(args.pdf, dpi=args.dpi)

    var_db = run_vlm_variables(pages)
    validate_with(VARIABLE_DB_SCHEMA, var_db)
    with open(args.var_out, "w", encoding="utf-8") as f:
        json.dump(var_db, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.var_out}")

    eq_db = run_vlm_equations(pages)
    validate_with(EQUATION_DB_SCHEMA, eq_db)
    with open(args.eq_out, "w", encoding="utf-8") as f:
        json.dump(eq_db, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.eq_out}")

    if args.summary_out:
        write_extraction_summary(args.summary_out, var_db, eq_db, args.var_out, args.eq_out)
        print(f"Wrote {args.summary_out}")

def run_deduplicator(items: List[EquationItem]) -> List[CanonicalEquation]:
    """簡易的な deduplicator。空白除去済み LaTeX をキーに canonical を作る。"""
    canon: Dict[str, CanonicalEquation] = {}
    for item in items:
        normalized = re.sub(r"\s+", "", item.latex)
        if not normalized:
            continue
        entry = canon.get(normalized)
        if entry is None:
            entry = CanonicalEquation(
                key=normalized,
                latex=item.latex,
                support_ids=[item.id],
                role=None,
            )
            canon[normalized] = entry
        else:
            entry.support_ids.append(item.id)
    return list(canon.values())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harmonize two equation-variable dataset pairs and append "
        "the results to the consolidated database."
    )
    parser.add_argument("--eqs-primary", required=True, type=Path)
    parser.add_argument("--vars-primary", required=True, type=Path)
    parser.add_argument("--eqs-secondary", required=True, type=Path)
    parser.add_argument("--vars-secondary", required=True, type=Path)
    parser.add_argument("--database-eqs", required=True, type=Path)
    parser.add_argument("--database-vars", required=True, type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    primary_eqs_data = load_json(args.eqs_primary)
    secondary_eqs_data = load_json(args.eqs_secondary)
    primary_vars_data = load_json(args.vars_primary)
    secondary_vars_data = load_json(args.vars_secondary)

    substitutions, remaining_secondary_vars, matched_pairs = harmonize_variables(
        primary_vars_data.get("variables", []),
        secondary_vars_data.get("variables", []),
    )

    harmonized_secondary_eqs = apply_symbol_mapping_to_equations(
        secondary_eqs_data.get("equations", []), substitutions
    )

    _, filtered_secondary_eqs = merge_equation_sets(
        primary_eqs_data.get("equations", []),
        harmonized_secondary_eqs,
    )

    database_eqs = load_json(args.database_eqs).get("equations", [])
    database_vars = load_json(args.database_vars).get("variables", [])

    merged_equations = deduplicate_equations(
        database_eqs,
        list(primary_eqs_data.get("equations", [])) + filtered_secondary_eqs,
    )
    merged_variables = deduplicate_variables(
        database_vars,
        list(primary_vars_data.get("variables", [])) + remaining_secondary_vars,
    )

    dump_json(args.database_eqs, {"equations": merged_equations})
    dump_json(args.database_vars, {"variables": merged_variables})

    if matched_pairs:
        print("Harmonized variables:")
        for source, target in matched_pairs:
            print(
                f"  {source.get('symbol')} ({source.get('name')}) -> "
                f"{target.get('symbol')} ({target.get('name')})"
            )

    print(
        f"Database updated with {len(primary_eqs_data.get('equations', []))} "
        f"primary equations, {len(filtered_secondary_eqs)} secondary equations, "
        f"{len(primary_vars_data.get('variables', []))} primary variables, and "
        f"{len(remaining_secondary_vars)} secondary variables."
    )





# =====================
# 3) compiler（LLM例）
# =====================
llm = ChatOpenAI(model="gpt-5", temperature=0)

def compile_model(canon: List[CanonicalEquation], meta: Dict[str, Any]) -> CompiledModel:
    """
    canonical_equations を受け取り、モデルJSONを合成。
    ここでは LLM に“構造化JSON”で出させ、Pydanticで検証するパターン。
    """
    prompt = f"""
あなたは化学・プロセス工学のモデリングのプロフェッショナルです.
今から数式情報を渡す.これらの数式を用いて物理モデルを作成してほしい．具体的には，C_A_0を用いてC_Cを表現する数式を作成してほしい．
- 形式：{{"title": ..., "entities": {{"variables": [...], "parameters": [...]}}, "equations": [...], "notes": ...}}
- json形式のみの出力


meta:
{meta}

equations:
{[c.model_dump() for c in canon]}
"""
    msg = llm.invoke([HumanMessage(content=prompt)])
    text = msg.content

    # JSON抽出（最も外側の{}を拾う簡易法）: 実運用では構造化出力(Structured Output)推奨
    import re, json
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("LLM出力からJSONを抽出できませんでした")
    data = json.loads(m.group(0))
    try:
        return CompiledModel(**data)
    except ValidationError as e:
        # 失敗時は最小骨組みで返す
        return CompiledModel(title=meta.get("title","model"), entities={}, equations=[], notes=f"validation error: {str(e)}")

# =====================
# 4) LangGraph ノード定義
# =====================
def extract_node(state: PipelineState) -> PipelineState:
    items = run_extractor(state["passages"], state.get("source_meta", {}))
    state["raw_equations"] = [it.model_dump() for it in items]
    state["logs"].append(f"extract: {len(items)} items")
    return state

def dedup_node(state: PipelineState) -> PipelineState:
    items = [EquationItem(**it) for it in state["raw_equations"]]
    canon = run_deduplicator(items)
    state["canonical_equations"] = [c.model_dump() for c in canon]
    state["logs"].append(f"dedup: {len(canon)} canonical")
    return state

def compile_node(state: PipelineState) -> PipelineState:
    canon = [CanonicalEquation(**c) for c in state["canonical_equations"]]
    model = compile_model(canon, state.get("source_meta", {}))
    state["compiled_model"] = model.model_dump()
    state["logs"].append("compile: done")
    return state

# =====================
# 5) グラフ構築＆実行
# =====================
def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("compile", compile_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "dedup")
    graph.add_edge("dedup", "compile")
    graph.add_edge("compile", END)

    memory = MemorySaver()  # 実験中のチェックポイント保存に便利
    return graph.compile(checkpointer=memory)

# =====================
# 6) 入口関数
# =====================
def run_pipeline(doc_id: str, passages: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    app = build_graph()
    init: PipelineState = {
        "doc_id": doc_id,
        "source_meta": meta,
        "passages": passages,
        "raw_equations": [],
        "canonical_equations": [],
        "compiled_model": {},
        "logs": []
    }
    final_state = app.invoke(
        init,
        config={"configurable": {"thread_id": doc_id or "default"}},
    )  # 同期実行（イベント逐次も可能）
    return {
        "model": final_state["compiled_model"],
        "equations_canonical": final_state["canonical_equations"],
        "logs": final_state["logs"]
    }


if __name__ == "__main__":
    extractor_cli()
