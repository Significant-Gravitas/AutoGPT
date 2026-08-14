export type OutputType = "table" | "doc" | "image" | "unknown";

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

/**
 * Reduce a run's outputs map to the single value the typed viewer renders,
 * mirroring the backend classifier: the first non-empty pin, collapsing a
 * single-value pin to that value and a multi-value pin to the list.
 */
export function pickPrimaryOutput(outputs: Record<string, unknown[]>): unknown {
  for (const values of Object.values(outputs)) {
    if (Array.isArray(values) && values.length > 0) {
      return values.length === 1 ? values[0] : values;
    }
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

export function toCsv(rows: Record<string, unknown>[]): string {
  const columns = tableColumns(rows);
  const escape = (value: unknown) => {
    const text = cellText(value);
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
