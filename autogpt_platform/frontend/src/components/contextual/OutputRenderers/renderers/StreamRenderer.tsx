import { FrameCornersIcon } from "@phosphor-icons/react";
import React, { useRef } from "react";

import { Button } from "@/components/atoms/Button/Button";

import {
  CopyContent,
  DownloadContent,
  OutputMetadata,
  OutputRenderer,
} from "../types";

// E2B Desktop exposes its live stream as a noVNC viewer served at `/vnc.html`.
// Matching on that path keeps this renderer scoped to desktop streams and lets
// every other URL fall through to the LinkRenderer.
const STREAM_PATH = "/vnc.html";

const IFRAME_SANDBOX =
  "allow-scripts allow-same-origin allow-forms allow-pointer-lock allow-popups";
const IFRAME_ALLOW = "clipboard-read; clipboard-write; fullscreen";

function isStreamUrl(value: unknown): value is string {
  if (typeof value !== "string") return false;
  try {
    const url = new URL(value.trim());
    return (
      (url.protocol === "https:" || url.protocol === "http:") &&
      url.pathname.endsWith(STREAM_PATH)
    );
  } catch {
    return false;
  }
}

function StreamFrame({ url }: { url: string }) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Native fullscreen on the iframe itself — the browser handles Esc to exit,
  // so there's no extra "expand" control duplicating the node's data viewer.
  function enterFullscreen() {
    iframeRef.current?.requestFullscreen?.();
  }

  return (
    <div className="relative w-full">
      <iframe
        ref={iframeRef}
        src={url}
        sandbox={IFRAME_SANDBOX}
        allow={IFRAME_ALLOW}
        className="aspect-video w-full rounded-lg border border-zinc-200 bg-black"
        title="Live desktop stream"
      />
      <div className="absolute right-2 top-2">
        <Button
          variant="secondary"
          size="small"
          onClick={enterFullscreen}
          className="h-fit min-w-0 gap-1.5 border border-zinc-200 p-2"
          aria-label="View full screen"
        >
          <FrameCornersIcon size={14} />
        </Button>
      </div>
    </div>
  );
}

function renderStream(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  return <StreamFrame url={String(value).trim()} />;
}

function getCopyContentStream(
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

function getDownloadContentStream(
  _value: unknown,
  _metadata?: OutputMetadata,
): DownloadContent | null {
  return null;
}

export const streamRenderer: OutputRenderer = {
  name: "StreamRenderer",
  // Above LinkRenderer (5) so stream URLs render as an embed, not a plain link.
  priority: 50,
  canRender: (value) => isStreamUrl(value),
  render: renderStream,
  getCopyContent: getCopyContentStream,
  getDownloadContent: getDownloadContentStream,
  isConcatenable: () => false,
};
