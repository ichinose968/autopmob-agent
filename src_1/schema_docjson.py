VARIABLE_DB_SCHEMA = {
  "name": "variable_db",
  "schema": {
    "type": "object",
    "properties": {
      "doc_id": {"type": "string"},
      "variables": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name":        {"type": "string"},   # 変数の“名称”（英語or原文、正規化を推奨）
            "symbol":      {"type": "string"},   # 記号（LaTeX可）例: "v", "\\dot{x}"
            "description": {"type": "string"},   # 説明文（1〜3文程度）
            "unit":        {"type": "string"},   # 単位（例: "m/s", "kg·m^2"）
            "page":        {"type": "integer"}   # 任意: 初出ページ番号
          },
          "required": ["name", "symbol", "description","unit"],
          "additionalProperties": False
        }
      },
      "provenance": { "type": "array", "items": { "type": "object" } }
    },
    "required": ["variables"],
    "additionalProperties": True
  }
}

# 追加: 数式DB（すべての数式の LaTeX と説明）
EQUATION_DB_SCHEMA = {
  "name": "equation_db",
  "schema": {
    "type": "object",
    "properties": {
      "doc_id": {"type": "string"},
      "equations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id":          {"type": "string"},   # 任意: 一意ID（"eq1" 等）
            "latex":       {"type": "string"},   # 必須: LaTeX 本文
            "description": {"type": "string"},   # 必須: 数式の意味/役割（1〜3文）
            "variables":   {
              "type": "array",
              "items": {"type": "string"}        # この数式に出る記号（"v","x","\\mu" 等）
            },
            "page":        {"type": "integer"}   # 任意: 掲載ページ番号
          },
          "required": ["latex", "description","variables"],
          "additionalProperties": False
        }
      },
      "provenance": { "type": "array", "items": { "type": "object" } }
    },
    "required": ["equations"],
    "additionalProperties": True
  }
}
