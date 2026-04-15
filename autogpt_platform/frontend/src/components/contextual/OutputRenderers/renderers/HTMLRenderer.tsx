import React from "react";
import {
  TAILWIND_CDN_URL,
  wrapWithHeadInjection,
} from "@/lib/iframe-sandbox-csp";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

function HTMLPreview({ value }: { value: string }) {
  // Inject Tailwind CDN — no CSP (see iframe-sandbox-csp.ts for why)
  const tailwindScript = `<script src="${TAILWIND_CDN_URL}"></script>`;
  const srcDoc = wrapWithHeadInjection(value, tailwindScript);
  return (
    <iframe
      sandbox="allow-scripts"
      srcDoc={srcDoc}
      className="h-96 w-full rounded border border-zinc-200"
      title="HTML preview"
    />
  );
}

function canRenderHTML(value: unknown, metadata?: OutputMetadata): boolean {
  if (typeof value !== "string") return false;
  if (metadata?.mimeType === "text/html") return true;
  const filename = metadata?.filename?.toLowerCase();
  if (filename?.endsWith(".html") || filename?.endsWith(".htm")) return true;
  return false;
}

function renderHTML(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  return <HTMLPreview value={String(value)} />;
}

function getCopyContentHTML(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const text = String(value);
  return {
    mimeType: "text/html",
    data: text,
    fallbackText: text,
    alternativeMimeTypes: ["text/plain"],
  };
}

function getDownloadContentHTML(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const text = String(value);
  return {
    data: new Blob([text], { type: "text/html" }),
    filename: metadata?.filename || "page.html",
    mimeType: "text/html",
  };
}

export const htmlRenderer: OutputRenderer = {
  name: "HTMLRenderer",
  priority: 42,
  canRender: canRenderHTML,
  render: renderHTML,
  getCopyContent: getCopyContentHTML,
  getDownloadContent: getDownloadContentHTML,
  isConcatenable: () => false,
};
