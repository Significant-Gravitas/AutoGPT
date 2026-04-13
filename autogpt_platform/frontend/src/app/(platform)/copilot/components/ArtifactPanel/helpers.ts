import {
  Code,
  File,
  FileHtml,
  FileText,
  Image,
  Table,
  VideoCamera,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

export interface ArtifactClassification {
  type:
    | "markdown"
    | "code"
    | "react"
    | "html"
    | "csv"
    | "json"
    | "image"
    | "video"
    | "pdf"
    | "text"
    | "download-only";
  icon: Icon;
  label: string;
  openable: boolean;
  hasSourceToggle: boolean;
}

const TEN_MB = 10 * 1024 * 1024;

// Catalog of classification kinds. Each entry defines the shared output
// shape; extension/MIME → kind mapping is handled by the lookup tables below.
const KIND: Record<string, ArtifactClassification> = {
  image: {
    type: "image",
    icon: Image,
    label: "Image",
    openable: true,
    hasSourceToggle: false,
  },
  video: {
    type: "video",
    icon: VideoCamera,
    label: "Video",
    openable: true,
    hasSourceToggle: false,
  },
  pdf: {
    type: "pdf",
    icon: FileText,
    label: "PDF",
    openable: true,
    hasSourceToggle: false,
  },
  csv: {
    type: "csv",
    icon: Table,
    label: "Spreadsheet",
    openable: true,
    hasSourceToggle: true,
  },
  html: {
    type: "html",
    icon: FileHtml,
    label: "HTML",
    openable: true,
    hasSourceToggle: true,
  },
  react: {
    type: "react",
    icon: FileHtml,
    label: "React",
    openable: true,
    hasSourceToggle: true,
  },
  markdown: {
    type: "markdown",
    icon: FileText,
    label: "Document",
    openable: true,
    hasSourceToggle: true,
  },
  json: {
    type: "json",
    icon: Code,
    label: "Data",
    openable: true,
    hasSourceToggle: true,
  },
  code: {
    type: "code",
    icon: Code,
    label: "Code",
    openable: true,
    hasSourceToggle: false,
  },
  text: {
    type: "text",
    icon: FileText,
    label: "Text",
    openable: true,
    hasSourceToggle: false,
  },
  "download-only": {
    type: "download-only",
    icon: File,
    label: "File",
    openable: false,
    hasSourceToggle: false,
  },
};

// Extension → kind. First match wins.
const EXT_KIND: Record<string, string> = {
  ".png": "image",
  ".jpg": "image",
  ".jpeg": "image",
  ".gif": "image",
  ".webp": "image",
  ".svg": "image",
  ".bmp": "image",
  ".ico": "image",
  ".avif": "image",
  ".mp4": "video",
  ".webm": "video",
  ".m4v": "video",
  ".pdf": "pdf",
  ".csv": "csv",
  ".tsv": "csv",
  ".html": "html",
  ".htm": "html",
  ".jsx": "react",
  ".tsx": "react",
  ".md": "markdown",
  ".mdx": "markdown",
  ".json": "json",
  ".jsonl": "code",
  ".txt": "text",
  ".log": "text",
  ".ics": "text",
  ".vcf": "text",
  ".env": "code",
  ".gitignore": "code",
  // code extensions
  ".js": "code",
  ".ts": "code",
  ".dart": "code",
  ".py": "code",
  ".rb": "code",
  ".go": "code",
  ".rs": "code",
  ".java": "code",
  ".c": "code",
  ".cpp": "code",
  ".h": "code",
  ".cs": "code",
  ".php": "code",
  ".swift": "code",
  ".kt": "code",
  ".sh": "code",
  ".bash": "code",
  ".zsh": "code",
  ".scss": "code",
  ".sass": "code",
  ".less": "code",
  ".graphql": "code",
  ".gql": "code",
  ".proto": "code",
  ".yml": "code",
  ".yaml": "code",
  ".toml": "code",
  ".ini": "code",
  ".cfg": "code",
  ".conf": "code",
  ".properties": "code",
  ".sql": "code",
  ".r": "code",
  ".lua": "code",
  ".pl": "code",
  ".scala": "code",
};

const EXACT_FILENAME_KIND: Record<string, string> = {
  dockerfile: "code",
  makefile: "code",
};

// Exact-match MIME → kind (fallback when extension doesn't match).
const MIME_KIND: Record<string, string> = {
  "application/pdf": "pdf",
  "text/csv": "csv",
  "text/tab-separated-values": "csv",
  "text/html": "html",
  "text/jsx": "react",
  "text/tsx": "react",
  "application/jsx": "react",
  "application/x-typescript-jsx": "react",
  "text/markdown": "markdown",
  "text/x-markdown": "markdown",
  "application/json": "json",
  "application/x-ndjson": "code",
  "application/ndjson": "code",
  "application/jsonl": "code",
  "application/javascript": "code",
  "text/javascript": "code",
  "application/typescript": "code",
  "text/typescript": "code",
  "application/xml": "code",
  "text/xml": "code",
};

const BINARY_MIMES = new Set([
  "application/zip",
  "application/x-zip-compressed",
  "application/gzip",
  "application/x-tar",
  "application/x-rar-compressed",
  "application/x-7z-compressed",
  "application/octet-stream",
  "application/wasm",
  "application/x-executable",
  "application/x-msdos-program",
  "application/vnd.microsoft.portable-executable",
]);

const PREVIEWABLE_IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "image/svg+xml",
  "image/bmp",
  "image/x-icon",
  "image/vnd.microsoft.icon",
  "image/avif",
]);

const PREVIEWABLE_VIDEO_MIMES = new Set([
  "video/mp4",
  "video/webm",
  "video/x-m4v",
]);

function getBasename(filename?: string): string {
  if (!filename) return "";
  const normalized = filename.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1]?.toLowerCase() ?? "";
}

function getExtension(filename?: string): string {
  if (!filename) return "";
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
}

export function classifyArtifact(
  mimeType: string | null,
  filename?: string,
  sizeBytes?: number,
): ArtifactClassification {
  // Size gate: >10MB is download-only regardless of type.
  if (sizeBytes && sizeBytes > TEN_MB) return KIND["download-only"];

  const basename = getBasename(filename);
  const exactKind = EXACT_FILENAME_KIND[basename];
  if (exactKind) return KIND[exactKind];

  if (basename === ".env" || basename.startsWith(".env.")) {
    return KIND.code;
  }

  // Extension first (more reliable than MIME for AI-generated files).
  const ext = getExtension(basename);
  const extKind = EXT_KIND[ext];
  if (extKind) return KIND[extKind];

  // MIME fallbacks.
  const mime = (mimeType ?? "").toLowerCase();
  if (PREVIEWABLE_IMAGE_MIMES.has(mime)) return KIND.image;
  if (PREVIEWABLE_VIDEO_MIMES.has(mime)) return KIND.video;
  const mimeKind = MIME_KIND[mime];
  if (mimeKind) return KIND[mimeKind];
  if (mime.startsWith("text/x-")) return KIND.code;
  if (
    mime.startsWith("image/") ||
    mime.startsWith("video/") ||
    mime.startsWith("font/")
  ) {
    return KIND["download-only"];
  }
  if (BINARY_MIMES.has(mime) || mime.startsWith("audio/")) {
    return KIND["download-only"];
  }
  if (mime.startsWith("text/")) return KIND.text;

  // Unknown extension + unknown MIME: don't open — we can't safely assume
  // this is text, and fetching a binary to dump it into a <pre> wastes
  // bandwidth and shows garbage.
  return KIND["download-only"];
}
