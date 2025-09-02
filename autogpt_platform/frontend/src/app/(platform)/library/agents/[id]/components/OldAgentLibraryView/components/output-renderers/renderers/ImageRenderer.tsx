import React from "react";
import { OutputRenderer, OutputMetadata, DownloadContent } from "../types";

export class ImageRenderer implements OutputRenderer {
  name = "ImageRenderer";
  priority = 40;

  private imageExtensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".ico",
  ];

  private imageMimeTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/svg+xml",
    "image/webp",
    "image/x-icon",
  ];

  canRender(value: any, metadata?: OutputMetadata): boolean {
    if (
      metadata?.type === "image" ||
      (metadata?.mimeType && this.imageMimeTypes.includes(metadata.mimeType))
    ) {
      return true;
    }

    if (typeof value === "string") {
      if (value.startsWith("data:image/")) {
        return true;
      }

      if (value.startsWith("http://") || value.startsWith("https://")) {
        return this.imageExtensions.some((ext) =>
          value.toLowerCase().includes(ext),
        );
      }

      if (metadata?.filename) {
        return this.imageExtensions.some((ext) =>
          metadata.filename!.toLowerCase().endsWith(ext),
        );
      }
    }

    return false;
  }

  render(value: any, metadata?: OutputMetadata): React.ReactNode {
    const imageUrl = String(value);
    const altText = metadata?.filename || "Output image";

    return (
      <div className="group relative">
        <img
          src={imageUrl}
          alt={altText}
          className="h-auto max-w-full rounded-md border border-gray-200"
          loading="lazy"
        />
      </div>
    );
  }

  getCopyContent(value: any, metadata?: OutputMetadata): string | null {
    return null;
  }

  getDownloadContent(
    value: any,
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

  isConcatenable(value: any, metadata?: OutputMetadata): boolean {
    return false;
  }
}
