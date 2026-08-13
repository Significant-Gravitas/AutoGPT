import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

const videoExtensions = [
  ".mp4",
  ".webm",
  ".ogg",
  ".mov",
  ".avi",
  ".mkv",
  ".m4v",
];

const videoMimeTypes = [
  "video/mp4",
  "video/webm",
  "video/ogg",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
];

function guessMimeType(url: string): string | null {
  if (url.startsWith("data:")) {
    const mimeMatch = url.match(/^data:([^;,]+)/);
    return mimeMatch?.[1] || null;
  }
  const extension = url.split(/[?#]/)[0].split(".").pop()?.toLowerCase();
  const mimeMap: Record<string, string> = {
    mp4: "video/mp4",
    webm: "video/webm",
    ogg: "video/ogg",
    mov: "video/quicktime",
    avi: "video/x-msvideo",
    mkv: "video/x-matroska",
    m4v: "video/mp4",
  };
  return extension ? mimeMap[extension] || null : null;
}

const YOUTUBE_REGEX =
  /^https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?.*v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;

const VIMEO_REGEX = /^https?:\/\/(?:www\.)?vimeo\.com\/(\d+)/;

function getYouTubeVideoID(url: string): string | null {
  const match = url.match(YOUTUBE_REGEX);
  return match ? match[1] : null;
}

function getVimeoVideoID(url: string): string | null {
  const match = url.match(VIMEO_REGEX);
  return match ? match[1] : null;
}

function isEmbeddableVideoURL(url: string): boolean {
  return getYouTubeVideoID(url) !== null || getVimeoVideoID(url) !== null;
}

function canRenderVideo(value: unknown, metadata?: OutputMetadata): boolean {
  if (typeof value !== "string") return false;

  if (
    metadata?.type === "video" ||
    (metadata?.mimeType && videoMimeTypes.includes(metadata.mimeType))
  ) {
    return true;
  }

  if (value.startsWith("data:video/")) {
    return true;
  }

  if (isEmbeddableVideoURL(value)) {
    return true;
  }

  if (value.startsWith("http://") || value.startsWith("https://")) {
    const cleanURL = value.split(/[?#]/)[0].toLowerCase();
    if (videoExtensions.some((ext) => cleanURL.endsWith(ext))) {
      return true;
    }
  }

  if (metadata?.filename) {
    return videoExtensions.some((ext) =>
      metadata.filename!.toLowerCase().endsWith(ext),
    );
  }

  return false;
}

function renderVideo(
  value: unknown,
  metadata?: OutputMetadata,
): React.ReactNode {
  const videoUrl = String(value);

  const youtubeID = getYouTubeVideoID(videoUrl);
  if (youtubeID) {
    return (
      <div className="group relative aspect-video w-full">
        <iframe
          className="h-full w-full rounded-md border border-gray-200"
          src={`https://www.youtube.com/embed/${youtubeID}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          sandbox="allow-scripts allow-same-origin allow-presentation allow-fullscreen allow-popups"
          title="YouTube video"
        />
      </div>
    );
  }

  const vimeoID = getVimeoVideoID(videoUrl);
  if (vimeoID) {
    return (
      <div className="group relative aspect-video w-full">
        <iframe
          className="h-full w-full rounded-md border border-gray-200"
          src={`https://player.vimeo.com/video/${vimeoID}`}
          allow="autoplay; fullscreen; picture-in-picture"
          allowFullScreen
          sandbox="allow-scripts allow-same-origin allow-presentation allow-fullscreen allow-popups"
          title="Vimeo video"
        />
      </div>
    );
  }

  return (
    <div className="group relative">
      <video
        controls
        className="h-auto max-w-full rounded-md border border-gray-200"
        preload="metadata"
      >
        <source src={videoUrl} type={metadata?.mimeType || "video/mp4"} />
        Your browser does not support the video tag.
      </video>
    </div>
  );
}

function getCopyContentVideo(
  value: unknown,
  metadata?: OutputMetadata,
): CopyContent | null {
  const videoUrl = String(value);

  // For embeddable URLs, just copy the URL text
  if (isEmbeddableVideoURL(videoUrl)) {
    return {
      mimeType: "text/plain",
      data: videoUrl,
      fallbackText: videoUrl,
    };
  }

  if (videoUrl.startsWith("data:")) {
    const mimeMatch = videoUrl.match(/data:([^;]+)/);
    const mimeType = mimeMatch?.[1] || "video/mp4";

    return {
      mimeType: mimeType,
      data: videoUrl,
      alternativeMimeTypes: ["text/plain"],
      fallbackText: videoUrl,
    };
  }

  const mimeType = metadata?.mimeType || guessMimeType(videoUrl) || "video/mp4";

  return {
    mimeType: mimeType,
    data: async () => {
      const response = await fetch(videoUrl);
      return await response.blob();
    },
    alternativeMimeTypes: ["text/plain"],
    fallbackText: videoUrl,
  };
}

function getDownloadContentVideo(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const videoUrl = String(value);

  if (isEmbeddableVideoURL(videoUrl)) {
    return {
      data: new Blob([videoUrl], { type: "text/plain" }),
      filename: metadata?.filename || "video-url.txt",
      mimeType: "text/plain",
    };
  }

  if (videoUrl.startsWith("data:")) {
    const [mimeInfo, base64Data] = videoUrl.split(",");
    const mimeType = mimeInfo.match(/data:([^;]+)/)?.[1] || "video/mp4";
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);

    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: mimeType });

    const extension = mimeType.split("/")[1] || "mp4";
    return {
      data: blob,
      filename: metadata?.filename || `video.${extension}`,
      mimeType,
    };
  }

  return {
    data: videoUrl,
    filename: metadata?.filename || "video.mp4",
    mimeType: metadata?.mimeType || "video/mp4",
  };
}

function isConcatenableVideo(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return false;
}

export const videoRenderer: OutputRenderer = {
  name: "VideoRenderer",
  priority: 45,
  canRender: canRenderVideo,
  render: renderVideo,
  getCopyContent: getCopyContentVideo,
  getDownloadContent: getDownloadContentVideo,
  isConcatenable: isConcatenableVideo,
};
