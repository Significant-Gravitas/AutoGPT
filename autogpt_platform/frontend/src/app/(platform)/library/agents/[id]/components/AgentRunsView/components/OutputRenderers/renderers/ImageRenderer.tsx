import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

const imageExtensions = [
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".bmp",
  ".svg",
  ".webp",
  ".ico",
];

const imageMimeTypes = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/bmp",
  "image/svg+xml",
  "image/webp",
  "image/x-icon",
];

function guessMimeType(url: string): string | null {
  const extension = url.split(".").pop()?.toLowerCase();
  const mimeMap: Record<string, string> = {
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    gif: "image/gif",
    bmp: "image/bmp",
    svg: "image/svg+xml",
    webp: "image/webp",
    ico: "image/x-icon",
  };
  return extension ? mimeMap[extension] || null : null;
}

function canRenderImage(value: unknown, metadata?: OutputMetadata): boolean {
  if (
    metadata?.type === "image" ||
    (metadata?.mimeType && imageMimeTypes.includes(metadata.mimeType))
  ) {
    return true;
  }

  if (typeof value === "object" && value !== null) {
    const obj = value as any;
    if (obj.url || obj.data || obj.path) {
      const urlOrData = obj.url || obj.data || obj.path;

      if (typeof urlOrData === "string") {
        if (urlOrData.startsWith("data:image/")) {
          return true;
        }

        if (
          urlOrData.startsWith("http://") ||
          urlOrData.startsWith("https://")
        ) {
          const hasImageExt = imageExtensions.some((ext) =>
            urlOrData.toLowerCase().includes(ext),
          );
          return hasImageExt;
        }
      }
    }

    if (obj.filename) {
      const hasImageExt = imageExtensions.some((ext) =>
        obj.filename.toLowerCase().endsWith(ext),
      );
      return hasImageExt;
    }
  }

  if (typeof value === "string") {
    if (value.startsWith("data:image/")) {
      return true;
    }

    if (value.startsWith("http://") || value.startsWith("https://")) {
      const hasImageExt = imageExtensions.some((ext) =>
        value.toLowerCase().includes(ext),
      );
      return hasImageExt;
    }

    if (metadata?.filename) {
      const hasImageExt = imageExtensions.some((ext) =>
        metadata.filename!.toLowerCase().endsWith(ext),
      );
      return hasImageExt;
    }
  }

  return false;
}

function renderImage(
  value: unknown,
  metadata?: OutputMetadata,
): React.ReactNode {
  const imageUrl = String(value);
  const altText = metadata?.filename || "Output image";

  return (
    <div className="group relative">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt={altText}
        className="h-auto max-w-full rounded-md border border-gray-200"
        loading="lazy"
      />
    </div>
  );
}

function getCopyContentImage(
  value: unknown,
  metadata?: OutputMetadata,
): CopyContent | null {
  const imageUrl = String(value);

  if (imageUrl.startsWith("data:")) {
    const mimeMatch = imageUrl.match(/data:([^;]+)/);
    const mimeType = mimeMatch?.[1] || "image/png";

    return {
      mimeType: mimeType,
      data: async () => {
        const response = await fetch(imageUrl);
        return await response.blob();
      },
      alternativeMimeTypes: ["image/png", "text/plain"],
      fallbackText: imageUrl,
    };
  }

  const mimeType = metadata?.mimeType || guessMimeType(imageUrl) || "image/png";

  return {
    mimeType: mimeType,
    data: async () => {
      const response = await fetch(imageUrl);
      return await response.blob();
    },
    alternativeMimeTypes: ["image/png", "text/plain"],
    fallbackText: imageUrl,
  };
}

function getDownloadContentImage(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const imageUrl = String(value);

  if (imageUrl.startsWith("data:")) {
    const [mimeInfo, base64Data] = imageUrl.split(",");
    const mimeType = mimeInfo.match(/data:([^;]+)/)?.[1] || "image/png";
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);

    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: mimeType });

    const extension = mimeType.split("/")[1] || "png";
    return {
      data: blob,
      filename: metadata?.filename || `image.${extension}`,
      mimeType,
    };
  }

  return {
    data: imageUrl,
    filename: metadata?.filename || "image.png",
    mimeType: metadata?.mimeType || "image/png",
  };
}

function isConcatenableImage(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return false;
}

export const imageRenderer: OutputRenderer = {
  name: "ImageRenderer",
  priority: 40,
  canRender: canRenderImage,
  render: renderImage,
  getCopyContent: getCopyContentImage,
  getDownloadContent: getDownloadContentImage,
  isConcatenable: isConcatenableImage,
};
