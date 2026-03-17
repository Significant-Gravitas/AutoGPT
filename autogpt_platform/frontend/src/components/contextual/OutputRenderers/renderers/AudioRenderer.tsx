import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

const audioExtensions = [".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"];

const audioMimeTypes = [
  "audio/mpeg",
  "audio/wav",
  "audio/ogg",
  "audio/mp4",
  "audio/aac",
  "audio/flac",
];

function guessMimeType(url: string): string | null {
  if (url.startsWith("data:")) {
    const mimeMatch = url.match(/^data:([^;,]+)/);
    return mimeMatch?.[1] || null;
  }
  const extension = url.split(/[?#]/)[0].split(".").pop()?.toLowerCase();
  const mimeMap: Record<string, string> = {
    mp3: "audio/mpeg",
    wav: "audio/wav",
    ogg: "audio/ogg",
    m4a: "audio/mp4",
    aac: "audio/aac",
    flac: "audio/flac",
  };
  return extension ? mimeMap[extension] || null : null;
}

function canRenderAudio(value: unknown, metadata?: OutputMetadata): boolean {
  if (typeof value !== "string") return false;

  if (
    metadata?.type === "audio" ||
    (metadata?.mimeType && audioMimeTypes.includes(metadata.mimeType))
  ) {
    return true;
  }

  if (value.startsWith("data:audio/")) {
    return true;
  }

  if (value.startsWith("http://") || value.startsWith("https://")) {
    const cleanURL = value.split(/[?#]/)[0].toLowerCase();
    if (audioExtensions.some((ext) => cleanURL.endsWith(ext))) {
      return true;
    }
  }

  if (metadata?.filename) {
    const cleanName = metadata.filename.toLowerCase();
    return audioExtensions.some((ext) => cleanName.endsWith(ext));
  }

  return false;
}

function renderAudio(
  value: unknown,
  metadata?: OutputMetadata,
): React.ReactNode {
  const audioURL = String(value);
  const mimeType =
    metadata?.mimeType || guessMimeType(audioURL) || "audio/mpeg";

  return (
    <div className="group relative">
      <audio controls preload="metadata" className="w-full">
        <source src={audioURL} type={mimeType} />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}

function getCopyContentAudio(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const audioURL = String(value);
  return {
    mimeType: "text/plain",
    data: audioURL,
    fallbackText: audioURL,
  };
}

const extensionMap: Record<string, string> = {
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
  "audio/aac": "aac",
  "audio/flac": "flac",
};

function getDownloadContentAudio(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const audioURL = String(value);
  const mimeType =
    metadata?.mimeType || guessMimeType(audioURL) || "audio/mpeg";
  const ext = extensionMap[mimeType] || "mp3";

  return {
    data: audioURL,
    filename: metadata?.filename || `audio.${ext}`,
    mimeType,
  };
}

function isConcatenableAudio(): boolean {
  return false;
}

export const audioRenderer: OutputRenderer = {
  name: "AudioRenderer",
  priority: 42,
  canRender: canRenderAudio,
  render: renderAudio,
  getCopyContent: getCopyContentAudio,
  getDownloadContent: getDownloadContentAudio,
  isConcatenable: isConcatenableAudio,
};
