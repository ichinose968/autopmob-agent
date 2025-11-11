from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from tap import Tap

logger = logging.getLogger(__name__)


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


class CLIArgs(Tap):
    """Typed CLI options for the model-builder workflow."""

    pdf: list[Path]
    objective_text: str | None = None
    objective_file: Path | None = None
    input_var: list[str] = []
    output_var: list[str] = []
    success_criteria: str | None = None
    max_models: int = 3
    model_name: str = "gpt-5-2025-08-07"
    temperature: float = 0.2

    def configure(self) -> None:
        self.description = "Run the PDF-to-model workflow driven by LangChain."

    def process_args(self) -> None:
        if not self.pdf:
            raise ValueError("At least one --pdf must be supplied.")


def _resolve_objective_text(args: CLIArgs) -> str:
    if args.objective_text:
        return args.objective_text
    if args.objective_file:
        return Path(args.objective_file).read_text(encoding="utf-8")
    raise ValueError("Provide either --objective-text or --objective-file.")


def _create_llm(model_name: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    _ = load_dotenv(find_dotenv())
    args = CLIArgs().parse_args()
    objective_text = _resolve_objective_text(args)
    documents = [
        PDFDocumentInput(path=Path(pdf_path)) for pdf_path in args.pdf
    ]
    objective = ObjectiveSpec(
        description=objective_text,
        input_variables=args.input_var,
        output_variables=args.output_var,
        success_criteria=args.success_criteria,
    )
    workflow_input = WorkflowInput(documents=documents, objective=objective)
    llm = _create_llm(args.model_name, args.temperature)
    workflow = ModelBuilderWorkflow(
        llm=llm,
        sample_variable_db=args.sample_var_db,
        sample_equation_db=args.sample_eq_db,
        max_models=args.max_models,
    )
    output = workflow.run(workflow_input)
    logger.info("Workflow finished, emitting JSON result")
    print(output.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
