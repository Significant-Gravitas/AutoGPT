import React from "react";
import { ArrowSquareOut } from "@phosphor-icons/react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

function canRenderLink(value: unknown, _metadata?: OutputMetadata): boolean {
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  return trimmed.startsWith("http://") || trimmed.startsWith("https://");
}

function getDisplayURL(url: string): string {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname === "/" ? "" : parsed.pathname;
    const display = parsed.hostname + path;
    return display.length > 60 ? display.slice(0, 57) + "..." : display;
  } catch {
    return url.length > 60 ? url.slice(0, 57) + "..." : url;
  }
}

function renderLink(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  const url = String(value).trim();
  const displayText = getDisplayURL(url);

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 rounded-md text-sm text-blue-600 underline decoration-blue-300 underline-offset-2 transition-colors hover:text-blue-800 hover:decoration-blue-500"
    >
      <span className="break-all">{displayText}</span>
      <ArrowSquareOut size={14} className="flex-shrink-0" />
    </a>
  );
}

function getCopyContentLink(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const url = String(value).trim();
  return {
    mimeType: "text/plain",
    data: url,
    fallbackText: url,
  };
}

function getDownloadContentLink(
  _value: unknown,
  _metadata?: OutputMetadata,
): DownloadContent | null {
  return null;
}

function isConcatenableLink(): boolean {
  return false;
}

export const linkRenderer: OutputRenderer = {
  name: "LinkRenderer",
  priority: 5,
  canRender: canRenderLink,
  render: renderLink,
  getCopyContent: getCopyContentLink,
  getDownloadContent: getDownloadContentLink,
  isConcatenable: isConcatenableLink,
};
