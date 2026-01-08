import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

function canRenderText(value: unknown, _metadata?: OutputMetadata): boolean {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function renderText(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  const textValue = String(value);

  return (
    <p className="resize-none overflow-x-auto whitespace-pre-wrap break-words border-none text-sm text-neutral-700">
      {textValue}
    </p>
  );
}

function getCopyContentText(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const textValue = String(value);
  return {
    mimeType: "text/plain",
    data: textValue,
    fallbackText: textValue,
  };
}

function getDownloadContentText(
  value: unknown,
  _metadata?: OutputMetadata,
): DownloadContent | null {
  const textValue = String(value);
  const blob = new Blob([textValue], { type: "text/plain" });

  return {
    data: blob,
    filename: _metadata?.filename || "output.txt",
    mimeType: "text/plain",
  };
}

function isConcatenableText(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return true;
}

export const textRenderer: OutputRenderer = {
  name: "TextRenderer",
  priority: 0,
  canRender: canRenderText,
  render: renderText,
  getCopyContent: getCopyContentText,
  getDownloadContent: getDownloadContentText,
  isConcatenable: isConcatenableText,
};
