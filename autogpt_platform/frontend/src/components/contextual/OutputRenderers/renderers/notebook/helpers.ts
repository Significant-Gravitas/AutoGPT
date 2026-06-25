import type { CopyContent, DownloadContent, OutputMetadata } from "../../types";
import type { Notebook, NotebookCell, NotebookOutput } from "./types";

export function joinSource(source: string | string[]): string {
  return Array.isArray(source) ? source.join("") : source;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function isSource(value: unknown): value is string | string[] {
  return typeof value === "string" || isStringArray(value);
}

function isExecutionCount(value: unknown): value is number | null {
  return value === null || typeof value === "number";
}

function isNotebookOutput(value: unknown): value is NotebookOutput {
  if (!isRecord(value)) return false;

  const outputType = value.output_type;
  if (
    outputType !== "stream" &&
    outputType !== "display_data" &&
    outputType !== "execute_result" &&
    outputType !== "error"
  ) {
    return false;
  }

  if (
    value.name !== undefined &&
    value.name !== "stdout" &&
    value.name !== "stderr"
  ) {
    return false;
  }

  if (value.text !== undefined && !isSource(value.text)) return false;

  if (
    value.data !== undefined &&
    (!isRecord(value.data) || !Object.values(value.data).every(isSource))
  ) {
    return false;
  }

  if (
    value.execution_count !== undefined &&
    !isExecutionCount(value.execution_count)
  ) {
    return false;
  }

  if (value.ename !== undefined && typeof value.ename !== "string") {
    return false;
  }
  if (value.evalue !== undefined && typeof value.evalue !== "string") {
    return false;
  }
  if (value.traceback !== undefined && !isStringArray(value.traceback)) {
    return false;
  }

  return true;
}

function isNotebookCell(value: unknown): value is NotebookCell {
  if (!isRecord(value)) return false;

  if (
    value.cell_type !== "code" &&
    value.cell_type !== "markdown" &&
    value.cell_type !== "raw"
  ) {
    return false;
  }

  if (!isSource(value.source)) return false;

  if (
    value.outputs !== undefined &&
    (!Array.isArray(value.outputs) || !value.outputs.every(isNotebookOutput))
  ) {
    return false;
  }

  if (
    value.execution_count !== undefined &&
    !isExecutionCount(value.execution_count)
  ) {
    return false;
  }

  if (value.metadata !== undefined && !isRecord(value.metadata)) return false;

  return true;
}

export function parseNotebook(value: unknown): Notebook | null {
  try {
    let obj: unknown = value;
    if (typeof value === "string") {
      obj = JSON.parse(value);
    }
    if (
      isRecord(obj) &&
      typeof obj.nbformat === "number" &&
      Array.isArray(obj.cells) &&
      obj.cells.every(isNotebookCell) &&
      (obj.metadata === undefined || isRecord(obj.metadata))
    ) {
      return obj as unknown as Notebook;
    }
  } catch {
    return null;
  }
  return null;
}

export function canRenderNotebook(
  value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return parseNotebook(value) !== null;
}

export function getCopyContentNotebook(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const raw =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return {
    mimeType: "application/json",
    data: raw,
    fallbackText: raw,
    alternativeMimeTypes: ["text/plain"],
  };
}

export function getDownloadContentNotebook(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const raw =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const blob = new Blob([raw], { type: "application/x-ipynb+json" });
  return {
    data: blob,
    filename: metadata?.filename ?? "notebook.ipynb",
    mimeType: "application/x-ipynb+json",
  };
}

export function isConcatenableNotebook(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return false;
}
