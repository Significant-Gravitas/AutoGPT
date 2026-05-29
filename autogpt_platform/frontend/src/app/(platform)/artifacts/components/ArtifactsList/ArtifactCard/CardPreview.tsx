"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useState } from "react";
import {
  deriveBadgeLabel,
  FileIllustration,
  pickFileTypeKey,
} from "../FileIllustration";
import {
  getFileDownloadUrl,
  getPreviewKind,
  type PreviewKind,
} from "../helpers";

interface Props {
  file: WorkspaceFileItem;
}

export function CardPreview({ file }: Props) {
  const kind = getPreviewKind(file.mime_type, file.size_bytes);

  return (
    <div className="relative aspect-[16/10] overflow-hidden border-b border-zinc-200 bg-zinc-100">
      <div className="absolute inset-x-6 bottom-0 top-6 origin-bottom overflow-hidden rounded-t-2xl bg-white shadow-md shadow-black/[0.08] ring-1 ring-black/5 transition-transform duration-300 ease-out group-hover:scale-[1.04] motion-reduce:transition-none motion-reduce:group-hover:scale-100">
        <PreviewBody file={file} kind={kind} />
      </div>
    </div>
  );
}

function PreviewBody({
  file,
  kind,
}: {
  file: WorkspaceFileItem;
  kind: PreviewKind;
}) {
  const [hasError, setHasError] = useState(false);
  const handleError = useCallback(() => setHasError(true), []);

  if (hasError || kind === "none") {
    return <Fallback file={file} />;
  }

  if (kind === "image") {
    return <ImagePreview file={file} onError={handleError} />;
  }

  if (kind === "video") {
    return <VideoPreview file={file} onError={handleError} />;
  }

  return <TextSnippetPreview file={file} onError={handleError} />;
}

function ImagePreview({
  file,
  onError,
}: {
  file: WorkspaceFileItem;
  onError: () => void;
}) {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={getFileDownloadUrl(file.id)}
        alt={file.name}
        loading="lazy"
        onLoad={() => setIsLoaded(true)}
        onError={onError}
        className={cn(
          "h-full w-full object-cover transition-opacity duration-300",
          isLoaded ? "opacity-100" : "opacity-0",
        )}
      />
    </>
  );
}

function VideoPreview({
  file,
  onError,
}: {
  file: WorkspaceFileItem;
  onError: () => void;
}) {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      <video
        src={getFileDownloadUrl(file.id)}
        preload="metadata"
        muted
        playsInline
        onLoadedData={() => setIsLoaded(true)}
        onError={onError}
        className={cn(
          "h-full w-full object-cover transition-opacity duration-300",
          isLoaded ? "opacity-100" : "opacity-0",
        )}
      />
    </>
  );
}

function Fallback({ file }: { file: WorkspaceFileItem }) {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <FileIllustration
        typeKey={pickFileTypeKey(file.mime_type)}
        label={deriveBadgeLabel(file.name, file.mime_type)}
        size="md"
      />
    </div>
  );
}

function LoadingPlaceholder({ file }: { file: WorkspaceFileItem }) {
  return (
    <div className="absolute inset-0">
      <Fallback file={file} />
      <ProgressBar />
    </div>
  );
}

function ProgressBar() {
  return (
    <div
      className="absolute inset-x-0 top-0 h-0.5 overflow-hidden bg-zinc-200/70"
      aria-hidden
    >
      <div className="h-full w-1/3 animate-progress-bar rounded-full bg-zinc-500" />
    </div>
  );
}

const TEXT_SNIPPET_CHARS = 1500;

function TextSnippetPreview({
  file,
  onError,
}: {
  file: WorkspaceFileItem;
  onError: () => void;
}) {
  const [text, setText] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(getFileDownloadUrl(file.id));
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const body = await res.text();
        if (!cancelled) setText(body.slice(0, TEXT_SNIPPET_CHARS));
      } catch {
        if (!cancelled) onError();
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [file.id, onError]);

  if (text === null) {
    return <LoadingPlaceholder file={file} />;
  }

  return (
    <pre className="h-full w-full overflow-hidden whitespace-pre-wrap break-words bg-white p-3 font-mono text-[10px] leading-[1.35] text-zinc-700">
      {text}
    </pre>
  );
}
