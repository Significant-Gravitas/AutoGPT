import {
  Code,
  File,
  FileHtml,
  FileText,
  Image,
  Table,
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
    | "pdf"
    | "text"
    | "download-only";
  icon: Icon;
  label: string;
  openable: boolean;
  hasSourceToggle: boolean;
}

const TEN_MB = 10 * 1024 * 1024;

const CODE_EXTENSIONS = new Set([
  ".js",
  ".ts",
  ".py",
  ".rb",
  ".go",
  ".rs",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".cs",
  ".php",
  ".swift",
  ".kt",
  ".sh",
  ".bash",
  ".zsh",
  ".yml",
  ".yaml",
  ".toml",
  ".ini",
  ".cfg",
  ".sql",
  ".r",
  ".lua",
  ".pl",
  ".scala",
]);

const REACT_EXTENSIONS = new Set([".jsx", ".tsx"]);

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".svg",
  ".bmp",
  ".ico",
]);

const BINARY_MIMES = new Set([
  "application/zip",
  "application/x-zip-compressed",
  "application/gzip",
  "application/x-tar",
  "application/x-rar-compressed",
  "application/x-7z-compressed",
  "application/octet-stream",
  "application/x-executable",
  "application/x-msdos-program",
  "application/vnd.microsoft.portable-executable",
]);

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
  const mime = mimeType?.toLowerCase() ?? "";
  const ext = getExtension(filename);

  // Size gate: >10MB is download-only
  if (sizeBytes && sizeBytes > TEN_MB) {
    return {
      type: "download-only",
      icon: File,
      label: "File",
      openable: false,
      hasSourceToggle: false,
    };
  }

  if (IMAGE_EXTENSIONS.has(ext)) {
    return {
      type: "image",
      icon: Image,
      label: "Image",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (ext === ".pdf") {
    return {
      type: "pdf",
      icon: FileText,
      label: "PDF",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (ext === ".csv") {
    return {
      type: "csv",
      icon: Table,
      label: "Spreadsheet",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (ext === ".html" || ext === ".htm") {
    return {
      type: "html",
      icon: FileHtml,
      label: "HTML",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (REACT_EXTENSIONS.has(ext)) {
    return {
      type: "react",
      icon: FileHtml,
      label: "React",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (ext === ".md" || ext === ".mdx") {
    return {
      type: "markdown",
      icon: FileText,
      label: "Document",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (ext === ".json") {
    return {
      type: "json",
      icon: Code,
      label: "Data",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (CODE_EXTENSIONS.has(ext)) {
    return {
      type: "code",
      icon: Code,
      label: "Code",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (ext === ".txt" || ext === ".log") {
    return {
      type: "text",
      icon: FileText,
      label: "Text",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (mime.startsWith("image/")) {
    return {
      type: "image",
      icon: Image,
      label: "Image",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (mime === "application/pdf") {
    return {
      type: "pdf",
      icon: FileText,
      label: "PDF",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (mime === "text/csv") {
    return {
      type: "csv",
      icon: Table,
      label: "Spreadsheet",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (mime === "text/html") {
    return {
      type: "html",
      icon: FileHtml,
      label: "HTML",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (
    mime === "text/jsx" ||
    mime === "text/tsx" ||
    mime === "application/jsx" ||
    mime === "application/x-typescript-jsx"
  ) {
    return {
      type: "react",
      icon: FileHtml,
      label: "React",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (mime === "text/markdown" || mime === "text/x-markdown") {
    return {
      type: "markdown",
      icon: FileText,
      label: "Document",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (mime === "application/json") {
    return {
      type: "json",
      icon: Code,
      label: "Data",
      openable: true,
      hasSourceToggle: true,
    };
  }

  if (
    mime.startsWith("text/x-") ||
    mime === "application/javascript" ||
    mime === "text/javascript" ||
    mime === "application/typescript" ||
    mime === "text/typescript" ||
    mime === "application/xml" ||
    mime === "text/xml"
  ) {
    return {
      type: "code",
      icon: Code,
      label: "Code",
      openable: true,
      hasSourceToggle: false,
    };
  }

  if (
    BINARY_MIMES.has(mime) ||
    mime.startsWith("audio/") ||
    mime.startsWith("video/")
  ) {
    return {
      type: "download-only",
      icon: File,
      label: "File",
      openable: false,
      hasSourceToggle: false,
    };
  }

  if (mime.startsWith("text/")) {
    return {
      type: "text",
      icon: FileText,
      label: "Text",
      openable: true,
      hasSourceToggle: false,
    };
  }

  return {
    type: "text",
    icon: File,
    label: "File",
    openable: true,
    hasSourceToggle: false,
  };
}
