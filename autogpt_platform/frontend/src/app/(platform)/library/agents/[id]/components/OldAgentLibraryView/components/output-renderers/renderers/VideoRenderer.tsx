import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

export class VideoRenderer implements OutputRenderer {
  name = "VideoRenderer";
  priority = 45;

  private videoExtensions = [
    ".mp4",
    ".webm",
    ".ogg",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
  ];

  private videoMimeTypes = [
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
  ];

  canRender(value: any, metadata?: OutputMetadata): boolean {
    if (
      metadata?.type === "video" ||
      (metadata?.mimeType && this.videoMimeTypes.includes(metadata.mimeType))
    ) {
      return true;
    }

    if (typeof value === "string") {
      if (value.startsWith("data:video/")) {
        return true;
      }

      if (value.startsWith("http://") || value.startsWith("https://")) {
        return this.videoExtensions.some((ext) =>
          value.toLowerCase().includes(ext),
        );
      }

      if (metadata?.filename) {
        return this.videoExtensions.some((ext) =>
          metadata.filename!.toLowerCase().endsWith(ext),
        );
      }
    }

    return false;
  }

  render(value: any, metadata?: OutputMetadata): React.ReactNode {
    const videoUrl = String(value);

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

  getCopyContent(value: any, metadata?: OutputMetadata): CopyContent | null {
    const videoUrl = String(value);

    // For data URLs, extract the MIME type
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

    // For URLs, use the metadata MIME type or guess from extension
    const mimeType =
      metadata?.mimeType || this.guessMimeType(videoUrl) || "video/mp4";

    return {
      mimeType: mimeType,
      data: async () => {
        // Fetch the video for copying
        const response = await fetch(videoUrl);
        return await response.blob();
      },
      alternativeMimeTypes: ["text/plain"],
      fallbackText: videoUrl,
    };
  }

  private guessMimeType(url: string): string | null {
    const extension = url.split(".").pop()?.toLowerCase();
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

  getDownloadContent(
    value: any,
    metadata?: OutputMetadata,
  ): DownloadContent | null {
    const videoUrl = String(value);

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

  isConcatenable(value: any, metadata?: OutputMetadata): boolean {
    return false;
  }
}
