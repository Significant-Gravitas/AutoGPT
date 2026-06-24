import type { Icon } from "@phosphor-icons/react";
import {
  BracketsCurlyIcon,
  CodeIcon,
  FileIcon,
  FilePdfIcon,
  FileTextIcon,
  ImageIcon,
  TableIcon,
  VideoCameraIcon,
} from "@phosphor-icons/react";

export type FileOrigin =
  | { kind: "session"; sessionId: string; href: string }
  | { kind: "builder"; href: string };

const SESSION_PATH_RE = /^\/sessions\/([^/]+)\//;

export function deriveFileOrigin(filePath: string | undefined): FileOrigin {
  const match = (filePath ?? "").match(SESSION_PATH_RE);
  if (match) {
    const sessionId = match[1];
    return {
      kind: "session",
      sessionId,
      href: `/copilot?sessionId=${encodeURIComponent(sessionId)}`,
    };
  }
  return { kind: "builder", href: "/build" };
}

export function getFileDownloadUrl(fileId: string): string {
  return `/api/proxy/api/workspace/files/${encodeURIComponent(fileId)}/download`;
}

// Fetches the file as a blob and triggers a browser download. Throws on a
// non-OK response so callers can surface the error (toast) and toggle their
// own loading state.
export async function downloadFileBlob(
  fileId: string,
  fileName: string,
): Promise<void> {
  const res = await fetch(getFileDownloadUrl(fileId));
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer revocation so browsers (Firefox/Edge) have time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function getFilePreviewUrl(
  fileId: string,
  opts: { width?: number; bytes?: number },
): string {
  const params = new URLSearchParams();
  if (opts.width) params.set("w", String(opts.width));
  if (opts.bytes) params.set("bytes", String(opts.bytes));
  const query = params.toString();
  const base = `/api/proxy/api/workspace/files/${encodeURIComponent(fileId)}/preview`;
  return query ? `${base}?${query}` : base;
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const exp = Math.min(
    BYTE_UNITS.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024)),
  );
  const value = bytes / Math.pow(1024, exp);
  const formatted = exp === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} ${BYTE_UNITS[exp]}`;
}

export function formatRelativeDate(input: string | Date): string {
  const date = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(date.getTime())) return "—";

  const diffMs = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year:
      date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

// Source-code extensions whose MIME type is ambiguous or actively misleading
// in the browser/OS database — most notably `.ts`, which resolves to
// `video/mp2t` (MPEG transport stream) and would otherwise render as a video.
// Classifying these by extension keeps source files out of the media path.
// Mirrors the copilot ArtifactPanel classifier, which is also extension-first.
const CODE_EXTENSIONS = new Set([
  "ts",
  "tsx",
  "js",
  "jsx",
  "mjs",
  "cjs",
  "mts",
  "cts",
  "py",
  "rb",
  "go",
  "rs",
  "java",
  "kt",
  "kts",
  "c",
  "h",
  "cpp",
  "cc",
  "hpp",
  "cs",
  "php",
  "swift",
  "scala",
  "sh",
  "bash",
  "zsh",
  "css",
  "scss",
  "sass",
  "less",
  "sql",
  "yaml",
  "yml",
  "toml",
  "ini",
  "vue",
  "svelte",
  "dart",
  "lua",
  "r",
]);

export function isCodeFile(fileName: string | undefined): boolean {
  const ext = (fileName ?? "").toLowerCase().match(/\.([a-z0-9]+)$/);
  return ext ? CODE_EXTENSIONS.has(ext[1]) : false;
}

export function getFileTypeLabel(
  mimeType: string | undefined,
  fileName?: string,
): string {
  if (isCodeFile(fileName)) return "Code";
  const mt = (mimeType ?? "").toLowerCase();
  if (mt.startsWith("image/")) return "Image";
  if (mt.startsWith("video/")) return "Video";
  if (mt.startsWith("audio/")) return "Audio";
  if (mt.includes("pdf")) return "PDF document";
  if (mt.includes("html") || mt.includes("xhtml")) return "Web page";
  if (mt.includes("csv")) return "Spreadsheet";
  if (mt.includes("spreadsheet") || mt.includes("excel")) return "Spreadsheet";
  if (mt.includes("json")) return "JSON data";
  if (mt.includes("markdown")) return "Markdown";
  if (mt.includes("text")) return "Document";
  return "Generated file";
}

export function getFileTypeIcon(
  mimeType: string | undefined,
  fileName?: string,
): Icon {
  if (isCodeFile(fileName)) return CodeIcon;
  const mt = (mimeType ?? "").toLowerCase();
  if (mt.startsWith("image/")) return ImageIcon;
  if (mt.startsWith("video/")) return VideoCameraIcon;
  if (mt.includes("pdf")) return FilePdfIcon;
  if (mt.includes("html") || mt.includes("xhtml")) return CodeIcon;
  if (
    mt.includes("csv") ||
    mt.includes("spreadsheet") ||
    mt.includes("excel")
  ) {
    return TableIcon;
  }
  if (mt.includes("json")) return BracketsCurlyIcon;
  if (mt.includes("text") || mt.includes("markdown")) return FileTextIcon;
  return FileIcon;
}

export type PreviewKind =
  | "image"
  | "video"
  | "pdf"
  | "office"
  | "csv"
  | "json"
  | "markdown"
  | "ics"
  | "vcard"
  | "text"
  | "none";

// These ceilings mirror the backend preview limits exactly, so we never fire
// a preview request for a file the backend would reject (it shows an
// illustration instead). Keep in sync with PREVIEW_MAX_* in backend preview.py.
const MAX_IMAGE_PREVIEW_BYTES = 10_000_000;
const MAX_VIDEO_PREVIEW_BYTES = 500_000_000;
const MAX_DOC_PREVIEW_BYTES = 50_000_000;
const MAX_TEXT_PREVIEW_BYTES = 50_000_000;
// ICS/VCF are parsed whole on the client, so cap the download size.
const MAX_STRUCTURED_PREVIEW_BYTES = 110_000;

const OFFICE_MIMES = new Set([
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);

function isOfficeFile(mime: string, name: string): boolean {
  return (
    OFFICE_MIMES.has(mime) ||
    name.endsWith(".pptx") ||
    name.endsWith(".docx") ||
    name.endsWith(".xlsx")
  );
}

export function getPreviewKind(
  mimeType: string | undefined,
  sizeBytes: number,
  fileName?: string,
): PreviewKind {
  const mt = (mimeType ?? "").toLowerCase();
  const name = (fileName ?? "").toLowerCase();

  // Extension-first for source code: `.ts` & friends carry misleading MIME
  // types (e.g. `video/mp2t`). Render them as a generic file card, not media.
  if (isCodeFile(name)) {
    return sizeBytes > MAX_TEXT_PREVIEW_BYTES ? "none" : "text";
  }

  if (mt.startsWith("image/") && !mt.includes("svg")) {
    return sizeBytes > MAX_IMAGE_PREVIEW_BYTES ? "none" : "image";
  }
  if (mt.startsWith("video/")) {
    return sizeBytes > MAX_VIDEO_PREVIEW_BYTES ? "none" : "video";
  }
  if (mt === "application/pdf" || name.endsWith(".pdf")) {
    return sizeBytes > MAX_DOC_PREVIEW_BYTES ? "none" : "pdf";
  }
  if (isOfficeFile(mt, name)) {
    return sizeBytes > MAX_DOC_PREVIEW_BYTES ? "none" : "office";
  }
  if (mt.includes("calendar") || name.endsWith(".ics")) {
    return sizeBytes > MAX_STRUCTURED_PREVIEW_BYTES ? "none" : "ics";
  }
  if (mt.includes("vcard") || name.endsWith(".vcf")) {
    return sizeBytes > MAX_STRUCTURED_PREVIEW_BYTES ? "none" : "vcard";
  }

  // csv/json/text fetch only the first few KB, but still skip absurdly large
  // files to match the backend's text ceiling.
  if (sizeBytes > MAX_TEXT_PREVIEW_BYTES) return "none";
  if (mt.includes("csv") || name.endsWith(".csv")) return "csv";
  if (mt.includes("json") || name.endsWith(".json")) return "json";
  if (
    mt.includes("markdown") ||
    name.endsWith(".md") ||
    name.endsWith(".markdown") ||
    name.endsWith(".mdx")
  ) {
    return "markdown";
  }
  if (
    mt.startsWith("text/") ||
    mt.includes("html") ||
    mt.includes("xhtml") ||
    mt.includes("xml") ||
    mt.includes("javascript") ||
    mt.includes("typescript") ||
    mt.includes("yaml")
  ) {
    return "text";
  }
  return "none";
}
