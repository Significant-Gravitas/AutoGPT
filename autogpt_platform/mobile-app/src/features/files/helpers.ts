import type { FileText, WorkspaceFile } from "@/lib/api/workspace";

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
// How much of a larger text file the preview route sends.
export const PREVIEW_BYTES = 131072;

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

/** A file bigger than the display limit is previewed, not downloaded whole. */
export function isPreviewed(file: Pick<WorkspaceFile, "size_bytes">) {
  return file.size_bytes > TEXT_DISPLAY_LIMIT;
}

/** Caps very long text so a large log can't freeze the viewer. */
export function capText({ text, isPreview }: FileText) {
  if (isPreview) return { text, cappedNotice: "Showing the first 128 KB." };
  if (text.length <= TEXT_DISPLAY_LIMIT) return { text, cappedNotice: null };
  return {
    text: text.slice(0, TEXT_DISPLAY_LIMIT),
    cappedNotice: "Showing the first 200 KB.",
  };
}

const REPLACEMENT_CHARACTER = 0xfffd;
const DECODE_CHUNK = 8192;

/** A preview is cut by bytes: drop a last character that lost its tail. */
export function trimPartialUtf8(bytes: Uint8Array) {
  let start = bytes.length;
  // A character has at most three continuation bytes after its lead byte.
  while (
    start > 0 &&
    bytes.length - start < 3 &&
    isContinuation(bytes[start - 1])
  ) {
    start -= 1;
  }
  if (start === 0) return bytes;
  const have = bytes.length - start + 1;
  return have < sequenceLength(bytes[start - 1])
    ? bytes.subarray(0, start - 1)
    : bytes;
}

/**
 * Hermes has no TextDecoder we can count on, and iOS fetch's text() gives up
 * on the whole body at one bad byte, so bad bytes become U+FFFD here instead.
 */
export function decodeUtf8(bytes: Uint8Array) {
  let text = "";
  let units: number[] = [];
  let index = 0;
  while (index < bytes.length) {
    const [codePoint, length] = readCodePoint(bytes, index);
    if (codePoint > 0xffff) {
      const offset = codePoint - 0x10000;
      units.push(0xd800 + (offset >> 10), 0xdc00 + (offset & 0x3ff));
    } else {
      units.push(codePoint);
    }
    index += length;
    if (units.length >= DECODE_CHUNK) {
      text += String.fromCharCode(...units);
      units = [];
    }
  }
  return text + String.fromCharCode(...units);
}

function isContinuation(byte: number) {
  return (byte & 0xc0) === 0x80;
}

function sequenceLength(lead: number) {
  if (lead >= 0xf0 && lead <= 0xf4) return 4;
  if (lead >= 0xe0) return lead <= 0xef ? 3 : 1;
  return lead >= 0xc2 ? 2 : 1;
}

// Follows the WHATWG decoder: a bad sequence becomes one U+FFFD and the byte
// that broke it starts the next read.
function readCodePoint(bytes: Uint8Array, index: number): [number, number] {
  const lead = bytes[index];
  if (lead < 0x80) return [lead, 1];
  const length = sequenceLength(lead);
  if (length === 1) return [REPLACEMENT_CHARACTER, 1];
  let lower = lead === 0xe0 ? 0xa0 : lead === 0xf0 ? 0x90 : 0x80;
  let upper = lead === 0xed ? 0x9f : lead === 0xf4 ? 0x8f : 0xbf;
  let codePoint = lead & (0xff >> (length + 1));
  for (let offset = 1; offset < length; offset += 1) {
    const byte = bytes[index + offset];
    if (byte === undefined || byte < lower || byte > upper) {
      return [REPLACEMENT_CHARACTER, offset];
    }
    codePoint = (codePoint << 6) | (byte & 0x3f);
    lower = 0x80;
    upper = 0xbf;
  }
  return [codePoint, length];
}
