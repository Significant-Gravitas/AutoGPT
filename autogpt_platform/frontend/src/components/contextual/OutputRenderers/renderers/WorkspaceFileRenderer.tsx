import { DownloadSimple, FileText } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

const imageMimeTypes = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/bmp",
  "image/svg+xml",
  "image/webp",
  "image/x-icon",
];

const videoMimeTypes = [
  "video/mp4",
  "video/webm",
  "video/ogg",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
];

const audioMimeTypes = [
  "audio/mpeg",
  "audio/ogg",
  "audio/wav",
  "audio/webm",
  "audio/aac",
  "audio/flac",
];

interface WorkspaceURI {
  fileID: string;
  mimeType: string | null;
}

function parseWorkspaceURI(value: string): WorkspaceURI | null {
  if (!value.startsWith("workspace://")) return null;
  const rest = value.slice("workspace://".length);
  const hashIndex = rest.indexOf("#");
  if (hashIndex === -1) {
    return { fileID: rest, mimeType: null };
  }
  return {
    fileID: rest.slice(0, hashIndex),
    mimeType: rest.slice(hashIndex + 1) || null,
  };
}

function buildDownloadURL(fileID: string): string {
  return `/api/proxy/api/workspace/files/${fileID}/download`;
}

function canRenderWorkspaceFile(value: unknown): boolean {
  return typeof value === "string" && value.startsWith("workspace://");
}

function getFileTypeLabel(mimeType: string | null): string {
  if (!mimeType) return "File";
  const sub = mimeType.split("/")[1];
  if (!sub) return "File";
  return `${sub.toUpperCase()} file`;
}

function renderWorkspaceFile(
  value: unknown,
  metadata?: OutputMetadata,
): ReactNode {
  const uri = parseWorkspaceURI(String(value));
  if (!uri) return null;

  const downloadURL = buildDownloadURL(uri.fileID);
  const mimeType = uri.mimeType || metadata?.mimeType || null;

  if (mimeType && imageMimeTypes.includes(mimeType)) {
    return (
      <div className="group relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={downloadURL}
          alt={metadata?.filename || "Image"}
          className="h-auto max-w-full rounded-md border border-gray-200"
          loading="lazy"
        />
      </div>
    );
  }

  if (mimeType && videoMimeTypes.includes(mimeType)) {
    return (
      <div className="group relative">
        <video
          controls
          className="h-auto max-w-full rounded-md border border-gray-200"
          preload="metadata"
        >
          <source src={downloadURL} type={mimeType} />
          Your browser does not support the video tag.
        </video>
      </div>
    );
  }

  if (mimeType && audioMimeTypes.includes(mimeType)) {
    return (
      <div className="group relative">
        <audio controls preload="metadata" className="w-full">
          <source src={downloadURL} type={mimeType} />
          Your browser does not support the audio tag.
        </audio>
      </div>
    );
  }

  // Generic file card with icon and download link
  const label = getFileTypeLabel(mimeType);
  return (
    <div className="flex items-center gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 dark:border-gray-700 dark:bg-gray-800">
      <FileText size={28} className="flex-shrink-0 text-gray-500" />
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
          {metadata?.filename || label}
        </span>
        {mimeType && (
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {mimeType}
          </span>
        )}
      </div>
      <a
        href={downloadURL}
        download
        className="flex-shrink-0 rounded-md p-1.5 text-gray-500 transition-colors hover:bg-gray-200 hover:text-gray-700 dark:hover:bg-gray-700 dark:hover:text-gray-300"
      >
        <DownloadSimple size={18} />
      </a>
    </div>
  );
}

function getCopyContentWorkspaceFile(
  value: unknown,
  metadata?: OutputMetadata,
): CopyContent | null {
  const uri = parseWorkspaceURI(String(value));
  if (!uri) return null;

  const downloadURL = buildDownloadURL(uri.fileID);
  const mimeType =
    uri.mimeType || metadata?.mimeType || "application/octet-stream";

  return {
    mimeType,
    data: async () => {
      const response = await fetch(downloadURL);
      return await response.blob();
    },
    alternativeMimeTypes: ["text/plain"],
    fallbackText: String(value),
  };
}

function getDownloadContentWorkspaceFile(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const uri = parseWorkspaceURI(String(value));
  if (!uri) return null;

  const mimeType =
    uri.mimeType || metadata?.mimeType || "application/octet-stream";
  const ext = mimeType.split("/")[1] || "bin";
  const filename = metadata?.filename || `file.${ext}`;

  return {
    data: buildDownloadURL(uri.fileID),
    filename,
    mimeType,
  };
}

function isConcatenableWorkspaceFile(): boolean {
  return false;
}

export const workspaceFileRenderer: OutputRenderer = {
  name: "WorkspaceFileRenderer",
  priority: 50, // Higher than video (45) and image (40) so it matches first
  canRender: canRenderWorkspaceFile,
  render: renderWorkspaceFile,
  getCopyContent: getCopyContentWorkspaceFile,
  getDownloadContent: getDownloadContentWorkspaceFile,
  isConcatenable: isConcatenableWorkspaceFile,
};
