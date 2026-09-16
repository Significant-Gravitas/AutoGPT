import React, { useRef } from "react";
import {
  ArrowExpandIcon,
  ArrowUpRight01Icon,
  ComputerIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

export interface DesktopStreamValue {
  kind: "desktop_stream";
  url: string;
  provider: string;
  sandbox_id: string;
  requires_auth?: boolean;
}

function isStreamUrl(url: unknown): url is string {
  if (typeof url !== "string") return false;
  // A root-relative path is our own proxy link; anything absolute must be
  // https, or a javascript: URL would run in our origin under
  // allow-scripts allow-same-origin.
  if (url.startsWith("/") && !url.startsWith("//")) return true;
  try {
    return new URL(url).protocol === "https:";
  } catch {
    return false;
  }
}

export function isDesktopStream(value: unknown): value is DesktopStreamValue {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  // This renderer is in the global registry, so any output shaped like a
  // stream reaches the iframe; only a URL that can resolve to our own origin
  // or to an https host may.
  return (
    candidate.kind === "desktop_stream" &&
    isStreamUrl(candidate.url) &&
    typeof candidate.sandbox_id === "string"
  );
}

interface PreviewProps {
  value: DesktopStreamValue;
  /** False on a shared transcript: the owner-bound link would only answer
   *  the viewer with a 404, so the frame is replaced by a notice. */
  ownerView?: boolean;
}

export function DesktopStreamPreview({
  value,
  ownerView = true,
}: PreviewProps) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const ownerOnly = value.requires_auth === true && !ownerView;

  function handleFullscreen() {
    frameRef.current?.requestFullscreen();
  }

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200">
      <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-50 px-3 py-2">
        <div className="flex items-center gap-2 text-sm text-zinc-700">
          <Icon icon={ComputerIcon} size={16} />
          <span className="font-medium">Interactive Desktop</span>
          <span className="rounded bg-zinc-200 px-1.5 py-0.5 text-xs uppercase text-zinc-600">
            {value.provider}
          </span>
        </div>
        {!ownerOnly && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleFullscreen}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-200"
              aria-label="Fullscreen"
            >
              <Icon icon={ArrowExpandIcon} size={14} />
              Fullscreen
            </button>
            <a
              href={value.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-200"
            >
              <Icon icon={ArrowUpRight01Icon} size={14} />
              Open in new tab
            </a>
          </div>
        )}
      </div>
      {ownerOnly ? (
        <div className="flex aspect-video w-full items-center justify-center bg-zinc-900 px-6 text-center text-sm text-zinc-400">
          The live desktop is only visible to the owner of this chat.
        </div>
      ) : (
        <iframe
          ref={frameRef}
          src={value.url}
          sandbox="allow-scripts allow-same-origin allow-popups"
          allow="clipboard-read; clipboard-write; fullscreen"
          className="aspect-video w-full bg-zinc-900"
          title={`Interactive desktop (${value.sandbox_id})`}
        />
      )}
      <p className="border-t border-zinc-200 bg-zinc-50 px-3 py-2 text-xs font-medium text-zinc-700">
        {value.requires_auth
          ? "Only the owner of this chat can view the live desktop. "
          : ""}
        The AI works on this desktop with full access, so anything signed in
        here is visible to it. Do not sign into personal accounts.
      </p>
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
