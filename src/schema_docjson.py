DOCJSON_SCHEMA = {
  "name": "docjson",
  "schema": {
    "type": "object",
    "properties": {
      "doc_id": {"type": "string"},
      "metadata": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "language_src": {"type": "string"},
          "language_norm": {"type": "string"},
          "domain_tags": {"type": "array", "items": {"type": "string"}},
          "doi": {"type": "string"}
        },
        "required": ["language_src", "language_norm", "domain_tags"]
      },
      "sections": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "spans": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "type": {"enum": ["equation", "citation", "note"]},
                  "latex": {"type": "string"},
                  "mathml": {"type": "string"},
                  "sympy": {"type": "string"},
                  "page": {"type": "integer"}
                },
                "required": ["type"]
              }
            }
          },
          "required": ["id","title","text"]
        }
      },
      "quantities": { "type": "array", "items": { "type": "object" } },
      "tables": { "type": "array", "items": { "type": "object" } },
      "figures": { "type": "array", "items": { "type": "object" } },
      "provenance": { "type": "array", "items": { "type": "object" } }
    },
    "required": ["metadata","sections","provenance"],
    "additionalProperties": True
  },
  "strict": True
}

