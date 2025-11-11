from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from tap import Tap

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gpt-5"
REPO_ROOT = Path(__file__).resolve().parent.parent


class CLIArgs(Tap):
    """Typed CLI options for the model-builder workflow."""

    config: Path
    results_dir: Path = Path("results")
    run_name: str | None = None

    def configure(self) -> None:
        self.description = "Run the PDF-to-model workflow driven by LangChain."

    def process_args(self) -> None:
        self.config = self.config.expanduser().resolve()
        self.results_dir = self.results_dir.expanduser().resolve()
        if not self.config.exists():
            raise FileNotFoundError(f"Config file not found: {self.config}")
        if self.config.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Config file must be a .yaml or .yml file.")


@dataclass
class RuntimeOptions:
    max_models: int
    model_name: str
    temperature: float


class PDFDocumentInput(BaseModel):
    path: Path
    doc_id: str | None = None

    @field_validator("path")
    def _ensure_pdf_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"PDF file not found: {value}")
        return value

    def resolved_id(self) -> str:
        if self.doc_id:
            return self.doc_id
        return re.sub(r"[^a-zA-Z0-9]+", "_", self.path.stem).lower()


class ObjectiveSpec(BaseModel):
    description: str
    input_variables: list[str] = Field(default_factory=list)
    output_variables: list[str] = Field(default_factory=list)
    success_criteria: str | None = None

    @field_validator("description")
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must not be empty")
        return value

    def as_prompt_fragment(self) -> str:
        lines = [f"Process goal: {self.description}"]
        if self.input_variables:
            lines.append("Measured inputs: " + ", ".join(self.input_variables))
        if self.output_variables:
            lines.append("Target outputs: " + ", ".join(self.output_variables))
        if self.success_criteria:
            lines.append(f"Success criteria: {self.success_criteria}")
        return "\n".join(lines)


class WorkflowInput(BaseModel):
    documents: list[PDFDocumentInput]
    objective: ObjectiveSpec


class ExtractedVariable(BaseModel):
    symbol: str
    definition: str
    description: str | None = None

    @field_validator("symbol", "definition")
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol/definition must not be empty")
        return value


class ExtractedEquation(BaseModel):
    latex: str
    variables: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("latex")
    def _ensure_latex_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Equation latex must not be empty")
        return value


class DocumentExtraction(BaseModel):
    document_id: str
    title: str | None = None
    narrative: str
    variables: list[ExtractedVariable] = Field(default_factory=list)
    equations: list[ExtractedEquation] = Field(default_factory=list)


class VariableEntry(BaseModel):
    variable_id: str
    symbol: str
    definition: str
    description: str | None = None
    documents: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)


class EquationEntry(BaseModel):
    equation_id: str
    latex: str
    variables: list[str] = Field(default_factory=list)
    description: str | None = None
    documents: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    summary: str


class OrganizedCorpus(BaseModel):
    documents: dict[str, DocumentRecord] = Field(default_factory=dict)
    variables: dict[str, VariableEntry] = Field(default_factory=dict)
    equations: dict[str, EquationEntry] = Field(default_factory=dict)


class EquationComponent(BaseModel):
    equation_id: str | None = None
    latex: str
    variables: list[str] = Field(default_factory=list)


class CandidateModel(BaseModel):
    name: str
    description: str
    variables: list[str] = Field(default_factory=list)
    equations: list[EquationComponent] = Field(default_factory=list)
    differentiator: str


class ModelBuilderResponse(BaseModel):
    models: list[CandidateModel]

    @model_validator(mode="after")
    def _ensure_models_present(self):
        if not self.models:
            raise ValueError(
                "The LLM must return at least one model candidate."
            )
        return self


class WorkflowOutput(BaseModel):
    organized_corpus: OrganizedCorpus
    models: list[CandidateModel]


def _load_config_data(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            "Config file must define a mapping/object at the top level."
        )
    return data


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(
        f"Expected string or list of strings, got {type(value).__name__}"
    )


def _prepare_workflow_from_config(
    config_data: dict[str, Any], config_path: Path
) -> tuple[WorkflowInput, RuntimeOptions]:
    base_dir = REPO_ROOT
    pdf_entries = config_data.get("pdf") or config_data.get("documents")
    if not pdf_entries:
        raise ValueError("Config must provide a `pdf` list.")

    documents: list[PDFDocumentInput] = []
    for entry in pdf_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each pdf entry must be a mapping with `path`.")
        if "path" not in entry:
            raise ValueError("PDF entry objects must include a `path` field.")
        doc_path = _resolve_path(entry["path"], base_dir)
        doc_id = entry.get("doc_id")
        documents.append(PDFDocumentInput(path=doc_path, doc_id=doc_id))

    objective_block = config_data.get("objective")
    if not isinstance(objective_block, dict):
        raise ValueError("Config must include an `objective` section.")
    objective_text = objective_block.get("description")
    if not objective_text:
        raise ValueError("`objective.description` is required in the config.")

    input_vars = _ensure_str_list(objective_block.get("input_variables"))
    output_vars = _ensure_str_list(objective_block.get("output_variables"))
    success_criteria = objective_block.get("success_criteria")

    objective = ObjectiveSpec(
        description=objective_text,
        input_variables=input_vars,
        output_variables=output_vars,
        success_criteria=success_criteria,
    )

    max_models = int(config_data.get("max_models", 3))
    if max_models < 1:
        raise ValueError("`max_models` must be >= 1.")
    model_name = config_data.get("model_name") or DEFAULT_MODEL_NAME
    temperature = float(config_data.get("temperature", 0.2))

    workflow_input = WorkflowInput(documents=documents, objective=objective)
    runtime_options = RuntimeOptions(
        max_models=max_models,
        model_name=model_name,
        temperature=temperature,
    )
    return workflow_input, runtime_options


def convert_pdf_to_base64(pdf_path):
    """
    Convert a PDF file to a Base64 encoded string.

    :param pdf_path: path to the pdf file
    :return: Base64 string
    """
    logger.debug("Encoding PDF '%s' to base64", pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_str = base64.b64encode(pdf_bytes).decode("utf-8")
    return pdf_str


class Extractor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _build_messages(
        self, objective: str, pdf_b64: str, pdf_file_path: Path
    ) -> list[BaseMessage]:
        messages = [
            SystemMessage(
                content="You are a scientific extraction agent. "
                "Given PDF you must list every variable and equation "
                "described in the paper. Return symbols and equations "
                "in LaTeX format. Always provide a title of the PDF and "
                "`narrative` that summarizes how the document describes "
                "the process as well as structured `variables` and "
                "`equations` lists.",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Objective:\n{objective}\n\n"},
                    {
                        "type": "file",
                        "file": {
                            "filename": f"{pdf_file_path}",
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                ]
            ),
        ]
        return messages

    async def _extract_single_async(
        self, doc: PDFDocumentInput, objective: ObjectiveSpec
    ) -> DocumentExtraction:
        pdf_b64 = convert_pdf_to_base64(doc.path)
        doc_id = doc.resolved_id()
        logger.info("Submitting document '%s' for extraction", doc_id)
        objective_text = objective.as_prompt_fragment()
        messages = self._build_messages(objective_text, pdf_b64, doc.path)
        structured_llm = self.llm.with_structured_output(DocumentExtraction)
        extraction = await structured_llm.ainvoke(messages)

        logger.info(
            "Extraction completed for '%s' (%d variables / %d equations)",
            doc_id,
            len(extraction.variables),  # type: ignore
            len(extraction.equations),  # type: ignore
        )
        return extraction.model_copy(  # type: ignore
            update={
                "document_id": doc_id,
            }
        )

    async def run_async(
        self, documents: Sequence[PDFDocumentInput], objective: ObjectiveSpec
    ) -> list[DocumentExtraction]:
        if not documents:
            logger.warning("Extractor run invoked with zero documents")
            return []

        logger.info(
            "Running extractor asynchronously on %d document(s)",
            len(documents),
        )
        tasks = [
            self._extract_single_async(doc, objective) for doc in documents
        ]

        results: list[DocumentExtraction] = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                logger.exception("Extraction failed for one document: %s", e)

        return results

    def run(
        self, documents: Sequence[PDFDocumentInput], objective: ObjectiveSpec
    ) -> list[DocumentExtraction]:
        return asyncio.run(self.run_async(documents, objective))


class DBOrganizer:
    def __init__(
        self,
    ):
        self.corpus = OrganizedCorpus()
        self.variable_index: dict[str, str] = {}
        self.equation_index: dict[str, str] = {}
        self.variable_counter = 1
        self.equation_counter = 1

    def ingest(self, extraction: DocumentExtraction) -> None:
        logger.info(
            "Ingesting document '%s' (%d variables / %d equations)",
            extraction.document_id,
            len(extraction.variables),
            len(extraction.equations),
        )
        self.corpus.documents[extraction.document_id] = DocumentRecord(
            doc_id=extraction.document_id,
            title=extraction.title or extraction.document_id,
            summary=extraction.narrative,
        )
        for variable in extraction.variables:
            self._upsert_variable(variable, extraction)
        for equation in extraction.equations:
            self._upsert_equation(equation, extraction)

    def export(self) -> OrganizedCorpus:
        logger.info(
            "Corpus snapshot: %d variables / %d equations",
            len(self.corpus.variables),
            len(self.corpus.equations),
        )
        return self.corpus

    def _allocate_variable_id(self) -> str:
        idx = f"var_{self.variable_counter:04d}"
        self.variable_counter += 1
        return idx

    def _allocate_equation_id(self) -> str:
        idx = f"eq_{self.equation_counter:04d}"
        self.equation_counter += 1
        return idx

    def _upsert_variable(
        self, variable: ExtractedVariable, extraction: DocumentExtraction
    ) -> None:
        variable_id = self.variable_index.get(variable.symbol)
        context = f"{extraction.document_id}: {variable.definition}"
        if variable_id is None:
            variable_id = self._allocate_variable_id()
            self.variable_index[variable.symbol] = variable_id
            self.corpus.variables[variable_id] = VariableEntry(
                variable_id=variable_id,
                symbol=variable.symbol,
                definition=variable.definition,
                description=variable.description,
                documents=[extraction.document_id],
                contexts=[context],
            )
            return

        entry = self.corpus.variables[variable_id]
        if extraction.document_id not in entry.documents:
            entry.documents.append(extraction.document_id)
        entry.contexts.append(context)
        if not entry.description and variable.description:
            entry.description = variable.description

    def _upsert_equation(
        self, equation: ExtractedEquation, extraction: DocumentExtraction
    ) -> None:
        equation_id = self.equation_index.get(equation.latex)
        context = (
            f"{extraction.document_id}: "
            f"{equation.description or 'equation from paper'}"
        )
        if equation_id is None:
            equation_id = self._allocate_equation_id()
            self.equation_index[equation.latex] = equation_id
            self.corpus.equations[equation_id] = EquationEntry(
                equation_id=equation_id,
                latex=equation.latex,
                variables=sorted(set(equation.variables)),
                description=equation.description,
                documents=[extraction.document_id],
                contexts=[context],
            )
            return

        entry = self.corpus.equations[equation_id]
        if extraction.document_id not in entry.documents:
            entry.documents.append(extraction.document_id)
        entry.contexts.append(context)
        merged_vars = set(entry.variables)
        merged_vars.update(equation.variables)
        entry.variables = sorted(merged_vars)
        if not entry.description and equation.description:
            entry.description = equation.description


class ModelBuilder:
    def __init__(self, llm: ChatOpenAI, max_models: int = 2):
        self.llm = llm
        self.max_models = max(1, max_models)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You assemble mathematical models by recombining "
                    "documented equations. Use only the given "
                    "variables/equations and always explain how each model "
                    "differs from the others.",
                ),
                (
                    "human",
                    "Objective:\n{objective}\n\n"
                    "Available database entries:\n{db_snapshot}\n\n"
                    "Enumerate {num_models} candidate models that satisfy the "
                    "objective. Each model must name the equations used, list "
                    "variables, and provide a differentiator sentence.",
                ),
            ]
        )

    def build(
        self, objective: ObjectiveSpec, corpus: OrganizedCorpus
    ) -> list[CandidateModel]:
        logger.info("Requesting up to %d candidate model(s)", self.max_models)
        snapshot = self._serialize_corpus(corpus)
        chain = self.prompt | self.llm.with_structured_output(
            ModelBuilderResponse
        )
        response = chain.invoke(
            {
                "objective": objective.as_prompt_fragment(),
                "db_snapshot": snapshot,
                "num_models": self.max_models,
            }
        )
        models = response.models  # type: ignore
        logger.info("LLM returned %d candidate model(s)", len(models))
        return models

    def _serialize_corpus(self, corpus: OrganizedCorpus) -> str:
        var_lines = [
            f"{e_.variable_id} ({e_.symbol}): {e_.definition}"
            for e_ in corpus.variables.values()
        ]
        eq_lines = [
            f"{e_.equation_id}: {e_.latex} | vars: {', '.join(e_.variables)}"
            for e_ in corpus.equations.values()
        ]
        doc_lines = [
            f"{doc_id}: {doc.title}"
            for doc_id, doc in corpus.documents.items()
        ]
        return "\n".join(
            [
                "Documents:",
                "\n".join(doc_lines) or "None",
                "\nVariables:",
                "\n".join(var_lines) or "None",
                "\nEquations:",
                "\n".join(eq_lines) or "None",
            ]
        )


class ModelBuilderWorkflow:
    def __init__(
        self,
        llm: ChatOpenAI,
        max_models: int = 2,
    ):
        self.extractor = Extractor(llm)
        self.organizer = DBOrganizer()
        self.builder = ModelBuilder(llm, max_models)

    def run(self, workflow_input: WorkflowInput) -> WorkflowOutput:
        logger.info(
            "Starting workflow for %d document(s)",
            len(workflow_input.documents),
        )
        extractions = self.extractor.run(
            workflow_input.documents, workflow_input.objective
        )
        for extraction in extractions:
            self.organizer.ingest(extraction)
        corpus = self.organizer.export()
        models = self.builder.build(workflow_input.objective, corpus)
        logger.info("Workflow completed with %d model(s)", len(models))
        return WorkflowOutput(organized_corpus=corpus, models=models)


def _create_llm(model_name: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)


def _prepare_run_directory(results_dir: Path, run_name: str | None) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        base_name = Path(run_name).stem
    else:
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    attempt = 0
    while True:
        suffix = f"_{attempt:02d}" if attempt else ""
        candidate = results_dir / f"{base_name}{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            logger.info("Created run directory %s", candidate)
            return candidate
        attempt += 1


def _persist_output(output: WorkflowOutput, run_dir: Path) -> Path:
    target_path = run_dir / "results.json"
    target_path.write_text(
        output.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved workflow output to %s", target_path)
    return target_path


def _copy_conditions_file(config_path: Path, run_dir: Path) -> Path:
    destination = run_dir / f"config_{config_path.name}"
    shutil.copy2(config_path, destination)
    logger.info("Copied condition file to %s", destination)
    return destination


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    _ = load_dotenv(find_dotenv())
    args = CLIArgs().parse_args()
    config_data = _load_config_data(args.config)
    workflow_input, runtime = _prepare_workflow_from_config(
        config_data, args.config
    )
    llm = _create_llm(runtime.model_name, runtime.temperature)
    workflow = ModelBuilderWorkflow(
        llm=llm,
        max_models=runtime.max_models,
    )
    output = workflow.run(workflow_input)
    run_dir = _prepare_run_directory(args.results_dir, args.run_name)
    result_path = _persist_output(output, run_dir)
    _copy_conditions_file(args.config, run_dir)
    logger.info(
        "Workflow finished; result stored at %s.",
        result_path,
    )


if __name__ == "__main__":
    main()
