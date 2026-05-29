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
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

export function getFileTypeLabel(mimeType: string | undefined): string {
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

export function getFileTypeIcon(mimeType: string | undefined): Icon {
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

export type PreviewKind = "image" | "video" | "text" | "none";

const MAX_TEXT_PREVIEW_BYTES = 200_000;
const MAX_IMAGE_PREVIEW_BYTES = 10_000_000;
const MAX_VIDEO_PREVIEW_BYTES = 500_000_000;

export function getPreviewKind(
  mimeType: string | undefined,
  sizeBytes: number,
): PreviewKind {
  const mt = (mimeType ?? "").toLowerCase();
  if (mt.startsWith("image/")) {
    return sizeBytes > MAX_IMAGE_PREVIEW_BYTES ? "none" : "image";
  }
  if (mt.startsWith("video/")) {
    return sizeBytes > MAX_VIDEO_PREVIEW_BYTES ? "none" : "video";
  }
  if (sizeBytes > MAX_TEXT_PREVIEW_BYTES) return "none";
  if (
    mt.startsWith("text/") ||
    mt.includes("html") ||
    mt.includes("xhtml") ||
    mt.includes("json") ||
    mt.includes("xml") ||
    mt.includes("csv") ||
    mt.includes("markdown") ||
    mt.includes("javascript") ||
    mt.includes("typescript") ||
    mt.includes("yaml")
  ) {
    return "text";
  }
  return "none";
}
