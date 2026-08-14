export type OutputType = "table" | "doc" | "image" | "unknown";

export const MAX_PREVIEW_ROWS = 100;
export const MAX_PREVIEW_COLUMNS = 20;

export function isOutputType(value: unknown): value is OutputType {
  return (
    value === "table" ||
    value === "doc" ||
    value === "image" ||
    value === "unknown"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asTableRows(value: unknown): Record<string, unknown>[] | null {
  if (Array.isArray(value) && value.length > 0 && value.every(isRecord)) {
    return value;
  }
  return null;
}

function collapsePin(values: unknown[]): unknown {
  return values.length === 1 ? values[0] : values;
}

function rendersAs(value: unknown, outputType: OutputType): boolean {
  if (outputType === "table") return asTableRows(value) !== null;
  if (outputType === "doc" || outputType === "image") {
    return typeof value === "string";
  }
  return false;
}

/**
 * Select the run-output value the typed viewer should render. The backend
 * classifier reports which pin it classified (`outputKey`) — prefer exactly
 * that pin. Without a key (legacy metadata), fall back to the first pin
 * whose value actually renders as the classified type, so a short status
 * string on an earlier pin can't shadow the classified table.
 */
export function pickOutputForType(
  outputs: Record<string, unknown[]>,
  outputType: OutputType,
  outputKey?: string | null,
): unknown {
  if (outputKey) {
    const values = outputs[outputKey];
    if (Array.isArray(values) && values.length > 0) {
      return collapsePin(values);
    }
  }
  for (const values of Object.values(outputs)) {
    if (!Array.isArray(values) || values.length === 0) continue;
    const value = collapsePin(values);
    if (rendersAs(value, outputType)) return value;
  }
  return null;
}

export function tableColumns(rows: Record<string, unknown>[]): string[] {
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) seen.add(key);
  }
  return Array.from(seen);
}

export function cellText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Leading whitespace/control characters followed by a formula trigger —
// Excel and LibreOffice both strip the leading junk before evaluating.
// eslint-disable-next-line no-control-regex
const FORMULA_PREFIX_RE = /^[\s\u0000-\u001f\u007f]*[=+\-@]/;

/**
 * Neutralize spreadsheet-formula injection: any untrusted *string* cell (or
 * header) that would be evaluated as a formula gets a leading apostrophe.
 * Genuine numbers and booleans pass through unprefixed so numeric columns
 * stay numeric (a negative number is not a formula).
 */
function neutralizeFormula(value: unknown, text: string): string {
  if (typeof value === "number" || typeof value === "boolean") return text;
  return FORMULA_PREFIX_RE.test(text) ? `'${text}` : text;
}

export function toCsv(
  rows: Record<string, unknown>[],
  columns: string[] = tableColumns(rows),
): string {
  const escape = (value: unknown) => {
    const text = neutralizeFormula(value, cellText(value));
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const header = columns.map(escape).join(",");
  const body = rows
    .map((row) => columns.map((column) => escape(row[column])).join(","))
    .join("\n");
  return `${header}\n${body}`;
}

export function downloadCsv(filename: string, csv: string): void {
  if (typeof document === "undefined") return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function buildRunLink(
  libraryAgentId: string | null | undefined,
  executionId: string,
): string | null {
  if (!libraryAgentId) return null;
  return (
    `/library/agents/${encodeURIComponent(libraryAgentId)}` +
    `?activeTab=runs&activeItem=${encodeURIComponent(executionId)}`
  );
}
