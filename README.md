# AutoPMoB Agent

AutoPMoB Agent turns technical PDFs into candidate process models by running a
three-stage LangChain + OpenAI workflow. Each stage is prompt-driven and
validated with Pydantic so that the final output is a consistent knowledge base
(variables, equations, document narratives) plus model proposals tied to a
specific engineering objective.

```
PDFs ──► Extractor ──► DBOrganizer ──► ModelBuilder ──► WorkflowOutput
           │              │                │
           │              │                └─ ranked model candidates
           │              └─ harmonized variable/equation graph
           └─ document narratives + raw math extractions
```

## Key Features
- **End-to-end automation** – ingest one or more PDFs, extract math objects, and
  assemble objective-aware model alternatives.
- **Prompt configurability** – each stage uses a dedicated YAML (system +
  human) template with placeholders such as `{objective}`, `{documents}`,
  `{db_snapshot}`, or `{num_models}`.
- **Structured guarantees** – responses are parsed into strongly typed models
  (Pydantic) so downstream tooling can rely on consistent JSON.
- **Traceable artifacts** – optional stage I/O dumps keep a copy of every LLM
  request/response for auditability.

See [README_JP.md](README_JP.md) for the Japanese edition.

---

## Repository Layout

```
autopmob-agent/
├── configs/          # Workflow configs (documents, objectives, prompt paths)
├── data/             # PDFs referenced by configs (place your own files here)
├── prompts/          # YAML prompt templates per stage (e.g., prompts/type_1/*)
├── results/          # Auto-created run outputs
├── scripts/
│   └── model_builder_agent.py  # Main workflow entry point
├── README.md
└── README_JP.md
```

---

## Prerequisites
- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)
- OpenAI API access with compatible `gpt-5` (or override via config)
- PDF files that describe the process/system you want to model

Set your credentials before running:

```bash
cp .env.example .env  # if you keep a template
echo "OPENAI_API_KEY=sk-..." >> .env
```

The script automatically loads `.env` via `python-dotenv`.

Install dependencies:

```bash
uv sync  # or `uv pip install -r requirements.txt` if you prefer pip tools
```

---

## Preparing a Workflow Config

All runs are driven by a YAML config under `configs/`. Start from
`configs/cstr-sample.yaml` and customize the following fields:

| Section | Required | Description |
|---------|----------|-------------|
| `pdf` | ✅ | List of `{path, doc_id?}` entries. Paths can be relative to the repo or absolute. `doc_id` is optional but helps track provenance. |
| `objective.description` | ✅ | Free-form text describing what the model should achieve. |
| `objective.input_variables` | optional | Measured inputs you expect to use. |
| `objective.output_variables` | optional | Controlled/observed outputs. |
| `objective.success_criteria` | optional | Additional guidance for the LLM. |
| `max_models` | optional (default 3) | Target number of model candidates. |
| `model_name` | optional | OpenAI chat model to call (defaults to `gpt-5`). |
| `temperature` | optional | LLM sampling temperature (float). |
| `prompts.extractor`, `prompts.organizer`, `prompts.model_builder` | ✅ | File path or inline mapping containing `system` and `human` strings. |

Minimal example:

```yaml
pdf:
  - path: data/example/paper.pdf
    doc_id: case-1

objective:
  description: >
    Build a nonlinear CSTR model capturing heat and mass balances.
  input_variables: [feed_flowrate, coolant_flowrate]
  output_variables: [reactor_temperature]
  success_criteria: >
    Provide at least two model structures and highlight assumptions.

max_models: 2
model_name: gpt-5
temperature: 0.2

prompts:
  extractor: prompts/type_1/extractor.yaml
  organizer: prompts/type_1/organizer.yaml
  model_builder: prompts/type_1/model_builder.yaml
```

### Prompt Format

Each prompt file contains two top-level keys:

```yaml
system: |-
  System-role instructions...
human: |-
  Human-role template using placeholders like {objective} or {documents}
```

Required placeholders:

- **Extractor** – `{objective}`, `{doc_id}`, `{title}` (the PDF itself is sent
  automatically as a file message).
- **DBOrganizer** – `{objective}`, `{documents}` (JSON string with per-doc
  narratives/extractions).
- **ModelBuilder** – `{objective}`, `{db_snapshot}`, `{num_models}`.

You can inline the YAML directly into the config (instead of referencing a
file) if needed.

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
| `--config PATH` | **Required.** Workflow config file. |
| `--results-dir PATH` | Target directory for runs (default `results/`). |
| `--run-name NAME` | Optional folder name. Timestamp-based if omitted. |
| `--save-stage-io` | When set, stores each stage's input/output into `artifacts/`. |

The script streams logs to stdout and prints the final `WorkflowOutput`
(`organized_corpus` + `models`) as JSON.

---

## Output Artifacts

Each run produces `results/<run_name>/`:

- `results.json` – canonical `WorkflowOutput`.
- `results/documents/*.json` – per-PDF breakdown: narratives, raw extractions,
  and references to harmonized graph entries.
- `results/models/models.json` – list of generated candidate models.
- `config_<original>.yaml` – exact config used for reproducibility.
- `artifacts/*.json` – optional stage dumps when `--save-stage-io` is enabled
  (`extractor_input.json`, `db_organizer_output.json`, etc.).

These files can be diffed across runs to evaluate prompt tweaks or document
changes.

---

## Development Notes

- Python dependencies are declared in `pyproject.toml`; use `uv sync` or your
  preferred workflow.
- The CLI lives in `scripts/model_builder_agent.py`. The pipeline is composed of
  `Extractor`, `DBOrganizer`, and `ModelBuilder` classes.
- Logging defaults to INFO. Set `LOGLEVEL=DEBUG` (or configure logging manually)
  for verbose traces.
- Run `uv run python scripts/model_builder_agent.py --help` for the latest CLI
  usage.

---

## Localization

The Japanese translation of this README is maintained at
[README_JP.md](README_JP.md). Keep both files in sync when making doc updates.
