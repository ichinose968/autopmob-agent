## AutoPMoB Agent

An experimental workflow for extracting mathematical relations from PDFs and assembling
candidate process models with LangChain + OpenAI. The core logic lives in
`scripts/model_builder_agent.py`, which orchestrates three LLM-driven stages:

1. **Extractor** – reads PDFs and returns narratives, variables, and equations.
2. **DBOrganizer** – harmonizes extracted variables/equations across documents.
3. **ModelBuilder** – combines the organized knowledge into objective-specific models.

Every stage is prompt-driven; prompts are stored as YAML files under `prompts/`.

---

## Repository Layout

```
autopmob-agent/
├── configs/          # Workflow configs (documents, objective, prompt paths)
├── data/             # PDFs or other inputs referenced by configs
├── prompts/          # YAML prompt templates (system + human per stage)
├── results/          # Run outputs (auto-created)
├── scripts/
│   └── model_builder_agent.py  # Main workflow
├── README.md
└── README_JP.md
```

---

## Prompt Files

Each stage uses a YAML file containing a `system` and `human` string:

```yaml
system: |-
  System-role text...
human: |-
  Human-role template with placeholders.
```

Placeholders such as `{objective}`, `{doc_id}`, `{title}`, `{documents}`,
`{db_snapshot}`, or `{num_models}` are injected by the workflow depending on stage.
Required placeholders per stage:

- **Extractor**: `{objective}`, `{doc_id}`, `{title}` (and the attachment is added automatically).
- **DBOrganizer**: `{objective}`, `{documents}`.
- **ModelBuilder**: `{objective}`, `{db_snapshot}`, `{num_models}`.

Example files:

```
prompts/
  extractor.yaml
  organizer.yaml
  model_builder.yaml
```

---

## Config Format

Create a YAML config in `configs/` (see `configs/cstr-sample.yaml`). Required fields:

```yaml
pdf:
  - path: data/.../paper1.pdf
    doc_id: cstr-1

objective:
  description: >
    Build a physical model for ...
  input_variables:
    - feed_flowrate
  output_variables:
    - reactor_temperature
  success_criteria: >
    Provide at least two candidate model structures...

max_models: 3
model_name: gpt-5
temperature: 0.2

prompts:
  extractor: prompts/extractor.yaml
  organizer: prompts/organizer.yaml
  model_builder: prompts/model_builder.yaml
```

Notes:
- PDF paths can be absolute or relative to the repo root (`~/Dropbox/.../autopmob-agent`).
- At least one document is required; `doc_id` is optional.
- Prompt paths are **mandatory**; the script does not ship with embedded defaults.

---

## Running the Workflow

```bash
uv run python scripts/model_builder_agent.py \
  --config configs/cstr-sample.yaml \
  --run-name cstr_run_001 \
  --save-stage-io
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config`      | **Required.** Path to the workflow YAML. |
| `--results-dir` | Parent directory for run folders (default `results/`). |
| `--run-name`    | Optional name for the run folder; timestamp is used if omitted. |
| `--save-stage-io` | When provided, saves each stage's input/output JSON under `artifacts/`. |

---

## Output Structure

Each run creates `results/<run_name>/` containing:

- `results/documents/*.json` – one file per PDF. Each file includes:
  - Document narrative summary.
  - Raw extraction for that document.
  - Harmonized variables/equations touching that document.
- `results/models/models.json` – candidate models returned by the ModelBuilder stage.
- `results.json` – index referencing the per-document files and models.
- `config_<original>.yaml` – copy of the config used for traceability.
- `artifacts/` (optional) – stage-by-stage inputs/outputs when `--save-stage-io` is set.

The aggregated `WorkflowOutput` (organized corpus + models) is still printed to stdout.

---

## Development Notes

- Python `>= 3.11` (see `pyproject.toml`).
- Dependencies are managed with [uv](https://github.com/astral-sh/uv) and declared in `pyproject.toml`.
- Key libraries: `langchain`, `langchain-openai`, `openai`, `PyYAML`.

---

## Japanese Documentation

A full Japanese translation is available in [README_JP.md](README_JP.md).
