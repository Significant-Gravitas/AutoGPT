import React from "react";
import { OutputRenderer, OutputMetadata, DownloadContent } from "../types";

export class TextRenderer implements OutputRenderer {
  name = "TextRenderer";
  priority = 0;

  canRender(value: any, metadata?: OutputMetadata): boolean {
    return (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    );
  }

  render(value: any, metadata?: OutputMetadata): React.ReactNode {
    const textValue = String(value);

    return (
      <p className="resize-none overflow-x-auto whitespace-pre-wrap break-words border-none text-sm text-neutral-700">
        {textValue}
      </p>
    );
  }

  getCopyContent(value: any, metadata?: OutputMetadata): string | null {
    return String(value);
  }

  getDownloadContent(
    value: any,
    metadata?: OutputMetadata,
  ): DownloadContent | null {
    const textValue = String(value);
    const blob = new Blob([textValue], { type: "text/plain" });

    return {
      data: blob,
      filename: metadata?.filename || "output.txt",
      mimeType: "text/plain",
    };
  }

  isConcatenable(value: any, metadata?: OutputMetadata): boolean {
    return true;
  }
}
