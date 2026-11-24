"use client";

import { lazyComponent } from "@/lib/lazy-component";
import React from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

// The markdown, math and highlighting libraries load when an output first
// shows; until they are here the output shows as plain text.
const markdownContent = lazyComponent(() =>
  import("./MarkdownContent").then((module) => module.MarkdownContent),
);

export const preloadMarkdownContent = markdownContent.preload;
const MarkdownContent = markdownContent.Loaded;

const markdownPatterns = [
  /```[\s\S]*?```/u, // Fenced code blocks (check first)
  /^#{1,6}\s+\S+/gmu, // ATX headers (require content)
  /\*\*[^*\n]+?\*\*/u, // **bold**
  /__(?!_)[^_\n]+?__(?!_)/u, // __bold__ (avoid ___/snake_case_)
  /(?<!\*)\*(?!\*)(?:[^*\n]|(?<=\\)\*)+?(?<!\\)\*(?!\*)/u, // *italic* (try to avoid **)
  /(?<!_)_(?!_)(?:[^_\n]|(?<=\\)_)+?(?<!\\)_(?!_)/u, // _italic_ with guards
  /\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/u, // Links with optional title (simple)
  /!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/u, // Images with optional title (simple)
  /`[^`\n]+`/u, // Inline code
  /^(?:\s*[-*+]\s+\S.*)(?:\n\s*[-*+]\s+\S.*)+$/gmu, // UL list (≥2 items)
  /^(?:\s*\d+\.\s+\S.*)(?:\n\s*\d+\.\s+\S.*)+$/gmu, // OL list (≥2 items)
  /^>\s+\S.*/gm, // Blockquotes
  /^\|[^|\n]+(\|[^|\n]+)+\|\s*$/gm, // Table row (at least two cells)
  /^\s*\|(?:\s*:?[-=]{3,}\s*\|)+\s*$/gm, // Table separator row
  /\$\$[\s\S]+?\$\$/u, // Display math
  /(?<!\\)(?<!\w)\$[^$\n]+?\$(?!\w)/u, // Inline math: avoid prices/ids
];

function canRenderMarkdown(value: unknown, metadata?: OutputMetadata): boolean {
  if (
    metadata?.type === "markdown" ||
    metadata?.mimeType === "text/markdown" ||
    metadata?.mimeType === "text/x-markdown"
  ) {
    return true;
  }

  if (typeof value !== "string") {
    return false;
  }

  if (metadata?.filename?.toLowerCase().endsWith(".md")) {
    return true;
  }

  let matchCount = 0;
  const requiredMatches = 2;

  for (const pattern of markdownPatterns) {
    if (pattern.test(value)) {
      matchCount++;
      if (matchCount >= requiredMatches) {
        return true;
      }
    }
  }

  return false;
}

function renderMarkdown(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  return <MarkdownOutput value={String(value)} />;
}

function MarkdownOutput({ value }: { value: string }) {
  if (!markdownContent.useIsLoaded()) {
    return (
      <div className="markdown-output break-words whitespace-pre-wrap">
        {value}
      </div>
    );
  }
  return <MarkdownContent value={value} />;
}

function getCopyContentMarkdown(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const markdownText = String(value);
  return {
    mimeType: "text/markdown",
    data: markdownText,
    fallbackText: markdownText,
    alternativeMimeTypes: ["text/plain"],
  };
}

function getDownloadContentMarkdown(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const markdownText = String(value);
  const blob = new Blob([markdownText], { type: "text/markdown" });

  return {
    data: blob,
    filename: metadata?.filename || "output.md",
    mimeType: "text/markdown",
  };
}

function isConcatenableMarkdown(
  _value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return true;
}

export const markdownRenderer: OutputRenderer = {
  name: "MarkdownRenderer",
  priority: 35,
  canRender: canRenderMarkdown,
  render: renderMarkdown,
  getCopyContent: getCopyContentMarkdown,
  getDownloadContent: getDownloadContentMarkdown,
  isConcatenable: isConcatenableMarkdown,
};
