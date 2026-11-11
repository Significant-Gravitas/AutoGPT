import type { WorkspaceFile } from "@/lib/api/workspace";

export type FileKind = "markdown" | "text" | "image" | "other";

// Workflow files the builder tools save as `agent.json` are plumbing, not
// something the user made, so the web leaves them out of file lists too.
const HIDDEN_FILE_NAMES = new Set(["agent.json"]);

// Older outputs have no purpose metadata. Only the reserved directory directly
// under the session root is a legacy fallback; nested user folders are not.
const TOOL_RESULT_PATH = /^\/sessions\/[^/]+\/tool-(?:results|outputs)\/[^/]+$/i;

const MARKDOWN_EXTENSIONS = new Set(["md", "markdown", "mdx"]);
const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp"]);
const TEXT_EXTENSIONS = new Set([
  "txt", "log", "ics", "vcf", "json", "jsonl", "csv", "tsv", "xml", "html",
  "htm", "yaml", "yml", "toml", "ini", "cfg", "conf", "properties", "env",
  "js", "jsx", "ts", "tsx", "py", "rb", "go", "rs", "java", "c", "cpp", "h",
  "cs", "php", "swift", "kt", "dart", "sh", "bash", "zsh", "css", "scss",
  "sass", "less", "graphql", "gql", "proto", "sql", "r", "lua", "pl", "scala",
]);
const IMAGE_MIMES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
const TEXT_MIMES = new Set([
  "application/json",
  "application/xml",
  "application/javascript",
  "application/typescript",
  "application/x-ndjson",
]);

const KIND_LABELS: Record<FileKind, string> = {
  markdown: "Markdown",
  text: "Text",
  image: "Image",
  other: "File",
};

export const TEXT_DISPLAY_LIMIT = 200 * 1024;

export function visibleFiles(files: WorkspaceFile[]) {
  return files.filter(
    (file) =>
      !HIDDEN_FILE_NAMES.has(file.name.toLowerCase()) &&
      !file.is_internal &&
      !isToolOutput(file),
  );
}

function isToolOutput(file: WorkspaceFile) {
  if (file.origin === "uploaded") return false;
  const purpose = file.metadata?.purpose;
  if (purpose !== undefined) return purpose === "tool-output";
  return TOOL_RESULT_PATH.test(file.path);
}

export function fileKind(file: Pick<WorkspaceFile, "name" | "mime_type">): FileKind {
  const extension = readExtension(file.name);
  if (MARKDOWN_EXTENSIONS.has(extension)) return "markdown";
  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (TEXT_EXTENSIONS.has(extension)) return "text";
  const mime = (file.mime_type ?? "").toLowerCase().split(";")[0].trim();
  if (mime === "text/markdown" || mime === "text/x-markdown") return "markdown";
  if (IMAGE_MIMES.has(mime)) return "image";
  if (mime.startsWith("text/") || TEXT_MIMES.has(mime)) return "text";
  return "other";
}

function readExtension(name: string) {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

export function kindLabel(kind: FileKind) {
  return KIND_LABELS[kind];
}

const SIZE_UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatFileSize(bytes: number) {
  if (!bytes || bytes < 0) return "0 B";
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    SIZE_UNITS.length - 1,
  );
  const value = bytes / Math.pow(1024, exponent);
  const rounded = exponent === 0 ? value : Math.round(value * 10) / 10;
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text} ${SIZE_UNITS[exponent]}`;
}

export function fileDetailLine(file: WorkspaceFile) {
  return `${formatFileSize(file.size_bytes)} · ${kindLabel(fileKind(file))}`;
}

/** Caps very long text so a large log can't freeze the viewer. */
export function capText(text: string) {
  if (text.length <= TEXT_DISPLAY_LIMIT) return { text, isCapped: false };
  return { text: text.slice(0, TEXT_DISPLAY_LIMIT), isCapped: true };
}
