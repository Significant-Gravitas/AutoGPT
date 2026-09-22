import JSZip from "jszip";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { ArtifactRef } from "../../../../store";

export function fileDownloadUrl(fileId: string): string {
  return `/api/proxy${getGetWorkspaceDownloadFileByIdUrl(fileId)}`;
}

export function isUploadedFile(item: WorkspaceFileItem): boolean {
  return item.origin === "uploaded";
}

export function fileItemToArtifactRef(item: WorkspaceFileItem): ArtifactRef {
  return {
    id: item.id,
    title: item.name,
    mimeType: item.mime_type ?? null,
    sourceUrl: fileDownloadUrl(item.id),
    origin: isUploadedFile(item) ? "user-upload" : "agent",
    sizeBytes: item.size_bytes ?? undefined,
  };
}

const UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatFileSize(bytes: number): string {
  if (!bytes || bytes < 0) return "0 B";
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    UNITS.length - 1,
  );
  const value = bytes / Math.pow(1024, exponent);
  const rounded = exponent === 0 ? value : Math.round(value * 10) / 10;
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text} ${UNITS[exponent]}`;
}

export function formatFileTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

interface ZipEntry {
  id: string;
  name: string;
}

interface DownloadZipDeps {
  fetchImpl?: (url: string) => Promise<Response>;
  save?: (blob: Blob, filename: string) => void;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function downloadFilesAsZip(
  entries: ZipEntry[],
  deps: DownloadZipDeps = {},
): Promise<void> {
  const fetchImpl = deps.fetchImpl ?? ((url: string) => fetch(url));
  const save = deps.save ?? triggerDownload;
  const zip = new JSZip();
  const used = new Set<string>();
  let added = 0;
  for (const entry of entries) {
    const res = await fetchImpl(fileDownloadUrl(entry.id));
    if (!res.ok) continue;
    let name = entry.name || entry.id;
    while (used.has(name)) name = `${entry.id.slice(0, 8)}-${name}`;
    used.add(name);
    zip.file(name, await res.blob());
    added++;
  }
  if (added === 0) {
    throw new Error("No files could be downloaded.");
  }
  const blob = await zip.generateAsync({ type: "blob" });
  save(blob, "workspace-files.zip");
}
