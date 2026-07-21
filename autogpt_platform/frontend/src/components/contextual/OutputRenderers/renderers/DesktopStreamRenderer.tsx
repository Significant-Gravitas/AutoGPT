import React, { useRef } from "react";
import {
  ArrowSquareOutIcon,
  CornersOutIcon,
  MonitorIcon,
} from "@phosphor-icons/react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

interface DesktopStreamValue {
  kind: "desktop_stream";
  url: string;
  provider: string;
  sandbox_id: string;
  requires_auth?: boolean;
}

function isDesktopStream(value: unknown): value is DesktopStreamValue {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.kind === "desktop_stream" &&
    typeof candidate.url === "string" &&
    typeof candidate.sandbox_id === "string"
  );
}

function DesktopStreamPreview({ value }: { value: DesktopStreamValue }) {
  const frameRef = useRef<HTMLIFrameElement>(null);

  function handleFullscreen() {
    frameRef.current?.requestFullscreen();
  }

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200">
      <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-50 px-3 py-2">
        <div className="flex items-center gap-2 text-sm text-zinc-700">
          <MonitorIcon size={16} />
          <span className="font-medium">Interactive Desktop</span>
          <span className="rounded bg-zinc-200 px-1.5 py-0.5 text-xs uppercase text-zinc-600">
            {value.provider}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleFullscreen}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-200"
            aria-label="Fullscreen"
          >
            <CornersOutIcon size={14} />
            Fullscreen
          </button>
          <a
            href={value.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-200"
          >
            <ArrowSquareOutIcon size={14} />
            Open in new tab
          </a>
        </div>
      </div>
      <iframe
        ref={frameRef}
        src={value.url}
        sandbox="allow-scripts allow-same-origin allow-popups"
        allow="clipboard-read; clipboard-write; fullscreen"
        className="aspect-video w-full bg-zinc-900"
        title={`Interactive desktop (${value.sandbox_id})`}
      />
    </div>
  );
}

function canRenderDesktopStream(
  value: unknown,
  _metadata?: OutputMetadata,
): boolean {
  return isDesktopStream(value);
}

function renderDesktopStream(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  if (!isDesktopStream(value)) return null;
  return <DesktopStreamPreview value={value} />;
}

function getCopyContentDesktopStream(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  if (!isDesktopStream(value)) return null;
  return {
    mimeType: "text/plain",
    data: value.url,
    fallbackText: value.url,
  };
}

function getDownloadContentDesktopStream(
  _value: unknown,
  _metadata?: OutputMetadata,
): DownloadContent | null {
  return null;
}

export const desktopStreamRenderer: OutputRenderer = {
  name: "DesktopStreamRenderer",
  priority: 80,
  canRender: canRenderDesktopStream,
  render: renderDesktopStream,
  getCopyContent: getCopyContentDesktopStream,
  getDownloadContent: getDownloadContentDesktopStream,
  isConcatenable: () => false,
};
