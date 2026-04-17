import { DownloadSimple, FileText } from "@phosphor-icons/react";
import { type ReactNode, useState } from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";
import { parseWorkspaceURI, isWorkspaceURI } from "@/lib/workspace-uri";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

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

function buildDownloadURL(fileID: string, shareToken?: string): string {
  if (shareToken) {
    return `/api/proxy/api/public/shared/${shareToken}/files/${fileID}/download`;
  }
  return `/api/proxy/api/workspace/files/${fileID}/download`;
}

function canRenderWorkspaceFile(value: unknown): boolean {
  return isWorkspaceURI(value);
}

function getFileTypeLabel(mimeType: string | null): string {
  if (!mimeType) return "File";
  const sub = mimeType.split("/")[1];
  if (!sub) return "File";
  return `${sub.toUpperCase()} file`;
}

function WorkspaceImage({ src, alt }: { src: string; alt: string }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="group relative">
      {!loaded && (
        <Skeleton className="absolute inset-0 h-full min-h-40 w-full rounded-md" />
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        className={`h-auto max-w-full rounded-md border border-gray-200 ${loaded ? "opacity-100" : "min-h-40 opacity-0"}`}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(true)}
      />
    </div>
  );
}

function WorkspaceVideo({ src, mimeType }: { src: string; mimeType: string }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="group relative">
      {!loaded && (
        <Skeleton className="absolute inset-0 h-full min-h-40 w-full rounded-md" />
      )}
      <video
        controls
        className={`h-auto max-w-full rounded-md border border-gray-200 ${loaded ? "opacity-100" : "min-h-40 opacity-0"}`}
        preload="metadata"
        onLoadedMetadata={() => setLoaded(true)}
        onError={() => setLoaded(true)}
      >
        <source src={src} type={mimeType} />
        Your browser does not support the video tag.
      </video>
    </div>
  );
}

function WorkspaceAudio({ src, mimeType }: { src: string; mimeType: string }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="group relative">
      {!loaded && (
        <Skeleton className="absolute inset-0 h-full min-h-12 w-full rounded-md" />
      )}
      <audio
        controls
        preload="metadata"
        className={`w-full ${loaded ? "opacity-100" : "min-h-12 opacity-0"}`}
        onLoadedMetadata={() => setLoaded(true)}
        onError={() => setLoaded(true)}
      >
        <source src={src} type={mimeType} />
        Your browser does not support the audio tag.
      </audio>
    </div>
  );
}

function renderWorkspaceFile(
  value: unknown,
  metadata?: OutputMetadata,
): ReactNode {
  const uri = parseWorkspaceURI(String(value));
  if (!uri) return null;

  const downloadURL = buildDownloadURL(uri.fileID, metadata?.shareToken);
  const mimeType = uri.mimeType || metadata?.mimeType || null;

  if (mimeType && imageMimeTypes.includes(mimeType)) {
    return (
      <WorkspaceImage src={downloadURL} alt={metadata?.filename || "Image"} />
    );
  }

  if (mimeType && videoMimeTypes.includes(mimeType)) {
    return <WorkspaceVideo src={downloadURL} mimeType={mimeType} />;
  }

  if (mimeType && audioMimeTypes.includes(mimeType)) {
    return <WorkspaceAudio src={downloadURL} mimeType={mimeType} />;
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

  const downloadURL = buildDownloadURL(uri.fileID, metadata?.shareToken);
  const mimeType =
    uri.mimeType || metadata?.mimeType || "application/octet-stream";

  return {
    mimeType,
    data: async () => {
      const response = await fetch(downloadURL);
      if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.status}`);
      }
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
    data: buildDownloadURL(uri.fileID, metadata?.shareToken),
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
