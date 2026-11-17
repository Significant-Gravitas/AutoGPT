import type { WorkspaceFile } from "@/lib/api/workspace";

// Ports of the web's file list helpers
// (frontend/src/app/(platform)/artifacts/components/ArtifactsList/helpers.ts).

export type FileTypeIcon =
  | "code"
  | "slides"
  | "document"
  | "table"
  | "image"
  | "video"
  | "pdf"
  | "json"
  | "other";

type FileInfo = Pick<WorkspaceFile, "name" | "mime_type" | "size_bytes">;

// `.ts` and friends carry misleading MIME types (`.ts` is `video/mp2t`), so
// source files are told apart by extension first.
const CODE_EXTENSIONS = new Set([
  "ts", "tsx", "js", "jsx", "mjs", "cjs", "mts", "cts", "py", "rb", "go", "rs",
  "java", "kt", "kts", "c", "h", "cpp", "cc", "hpp", "cs", "php", "swift",
  "scala", "sh", "bash", "zsh", "css", "scss", "sass", "less", "sql", "yaml",
  "yml", "toml", "ini", "vue", "svelte", "dart", "lua", "r",
]);

// The backend's ceiling for an image preview (PREVIEW_MAX_* in preview.py).
const MAX_IMAGE_PREVIEW_BYTES = 10_000_000;

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

function isCodeFile(name: string) {
  const extension = name.toLowerCase().match(/\.([a-z0-9]+)$/);
  return extension ? CODE_EXTENSIONS.has(extension[1]) : false;
}

function officeIcon(mime: string, name: string): FileTypeIcon | null {
  if (/\.pptx?$/.test(name)) return "slides";
  if (/\.docx?$/.test(name)) return "document";
  if (/\.xlsx?$/.test(name)) return "table";
  if (mime.includes("presentationml") || mime.includes("powerpoint")) return "slides";
  if (mime.includes("wordprocessingml") || mime.includes("msword")) return "document";
  return null;
}

export function fileTypeIcon({ name, mime_type }: Pick<FileInfo, "name" | "mime_type">): FileTypeIcon {
  if (isCodeFile(name)) return "code";
  const mime = (mime_type ?? "").toLowerCase();
  const office = officeIcon(mime, name.toLowerCase());
  if (office) return office;
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";
  if (mime.includes("pdf")) return "pdf";
  if (mime.includes("html") || mime.includes("xhtml")) return "code";
  if (mime.includes("csv") || mime.includes("spreadsheet") || mime.includes("excel")) {
    return "table";
  }
  if (mime.includes("json")) return "json";
  if (mime.includes("text") || mime.includes("markdown")) return "document";
  return "other";
}

/** Only images get a real thumbnail; the web also renders PDFs and slides. */
export function hasImageThumbnail({ name, mime_type, size_bytes }: FileInfo) {
  if (isCodeFile(name)) return false;
  const mime = (mime_type ?? "").toLowerCase();
  return (
    mime.startsWith("image/") &&
    !mime.includes("svg") &&
    size_bytes <= MAX_IMAGE_PREVIEW_BYTES
  );
}

export function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const exponent = Math.min(
    BYTE_UNITS.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024)),
  );
  const value = bytes / Math.pow(1024, exponent);
  return `${exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${BYTE_UNITS[exponent]}`;
}

export function formatDayLabel(input: string | Date, now = new Date()) {
  const date = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(date.getTime())) return "—";
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfDay = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const dayDiff = Math.round((startOfToday - startOfDay) / 86_400_000);
  if (dayDiff === 0) return "Today";
  if (dayDiff === 1) return "Yesterday";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

export function fileDetailLine(file: Pick<WorkspaceFile, "created_at" | "size_bytes">) {
  return `${formatDayLabel(file.created_at)} · ${formatFileSize(file.size_bytes)}`;
}

/**
 * This chat's files as the web orders them (made here, then attached). In an
 * expert's chat the expert's documents get their own list, so they are left
 * out here and no file shows twice.
 */
export function chatFiles(
  sessionFiles: WorkspaceFile[],
  expertDocuments: WorkspaceFile[] | null,
) {
  const listed = new Set((expertDocuments ?? []).map((file) => file.id));
  const remaining = sessionFiles.filter((file) => !listed.has(file.id));
  return [
    ...remaining.filter((file) => file.origin !== "uploaded"),
    ...remaining.filter((file) => file.origin === "uploaded"),
  ];
}
