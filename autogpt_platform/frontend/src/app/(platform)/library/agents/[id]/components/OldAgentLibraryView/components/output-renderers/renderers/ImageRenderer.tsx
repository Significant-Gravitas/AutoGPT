import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

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
    console.log("ImageRenderer.canRender checking:", {
      value,
      metadata,
      valueType: typeof value,
    });

    if (
      metadata?.type === "image" ||
      (metadata?.mimeType && this.imageMimeTypes.includes(metadata.mimeType))
    ) {
      console.log("ImageRenderer: Matched by metadata type/mimeType");
      return true;
    }

    // Check if value is an object with url/data property (common for file uploads)
    if (typeof value === "object" && value !== null) {
      if (value.url || value.data || value.path) {
        const urlOrData = value.url || value.data || value.path;
        console.log(
          "ImageRenderer: Found object with url/data/path:",
          urlOrData,
        );

        if (typeof urlOrData === "string") {
          if (urlOrData.startsWith("data:image/")) {
            console.log("ImageRenderer: Matched data URL");
            return true;
          }

          if (
            urlOrData.startsWith("http://") ||
            urlOrData.startsWith("https://")
          ) {
            const hasImageExt = this.imageExtensions.some((ext) =>
              urlOrData.toLowerCase().includes(ext),
            );
            console.log("ImageRenderer: URL has image extension:", hasImageExt);
            return hasImageExt;
          }
        }
      }

      // Check filename in object
      if (value.filename) {
        const hasImageExt = this.imageExtensions.some((ext) =>
          value.filename.toLowerCase().endsWith(ext),
        );
        console.log(
          "ImageRenderer: Object filename has image extension:",
          hasImageExt,
        );
        return hasImageExt;
      }
    }

    if (typeof value === "string") {
      if (value.startsWith("data:image/")) {
        console.log("ImageRenderer: String is data URL");
        return true;
      }

      if (value.startsWith("http://") || value.startsWith("https://")) {
        const hasImageExt = this.imageExtensions.some((ext) =>
          value.toLowerCase().includes(ext),
        );
        console.log(
          "ImageRenderer: String URL has image extension:",
          hasImageExt,
        );
        return hasImageExt;
      }

      if (metadata?.filename) {
        const hasImageExt = this.imageExtensions.some((ext) =>
          metadata.filename!.toLowerCase().endsWith(ext),
        );
        console.log(
          "ImageRenderer: Metadata filename has image extension:",
          hasImageExt,
        );
        return hasImageExt;
      }
    }

    console.log("ImageRenderer: No match found");
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

  getCopyContent(value: any, metadata?: OutputMetadata): CopyContent | null {
    const imageUrl = String(value);

    // For data URLs, extract the actual MIME type
    if (imageUrl.startsWith("data:")) {
      const mimeMatch = imageUrl.match(/data:([^;]+)/);
      const mimeType = mimeMatch?.[1] || "image/png";

      return {
        mimeType: mimeType,
        data: async () => {
          // Convert data URL to blob
          const response = await fetch(imageUrl);
          return await response.blob();
        },
        alternativeMimeTypes: ["image/png", "text/plain"],
        fallbackText: imageUrl,
      };
    }

    // For URLs, determine MIME type from metadata or extension
    const mimeType =
      metadata?.mimeType || this.guessMimeType(imageUrl) || "image/png";

    return {
      mimeType: mimeType,
      data: async () => {
        // Fetch the image
        const response = await fetch(imageUrl);
        return await response.blob();
      },
      alternativeMimeTypes: ["image/png", "text/plain"],
      fallbackText: imageUrl,
    };
  }

  private guessMimeType(url: string): string | null {
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
