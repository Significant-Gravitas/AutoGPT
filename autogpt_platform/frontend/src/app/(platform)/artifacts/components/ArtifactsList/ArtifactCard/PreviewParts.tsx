"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useEffect, useState } from "react";
import { fetchWorkspaceFileResource } from "../helpers";
import {
  deriveBadgeLabel,
  FileIllustration,
  pickFileTypeKey,
} from "../FileIllustration";

export function useFileText(
  file: WorkspaceFileItem,
  url: string,
  onError: () => void,
): string | null {
  const [text, setText] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setText(null);
    async function load() {
      try {
        const res = await fetchWorkspaceFileResource(file, url);
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const body = await res.text();
        if (!cancelled) setText(body);
      } catch {
        if (!cancelled) onError();
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [file, url, onError]);

  return text;
}

export function useFileBlobUrl(
  file: WorkspaceFileItem,
  url: string,
  onError: () => void,
): string | null {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setBlobUrl(null);
    async function load() {
      try {
        const res = await fetchWorkspaceFileResource(file, url);
        if (!res.ok) throw new Error(`Status ${res.status}`);
        objectUrl = URL.createObjectURL(await res.blob());
        if (!cancelled) setBlobUrl(objectUrl);
      } catch {
        if (!cancelled) onError();
      }
    }
    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [file, url, onError]);

  return blobUrl;
}

export function Fallback({ file }: { file: WorkspaceFileItem }) {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <FileIllustration
        typeKey={pickFileTypeKey(file.mime_type, file.name)}
        label={deriveBadgeLabel(file.name, file.mime_type)}
        size="md"
      />
    </div>
  );
}

export function LoadingPlaceholder({ file }: { file: WorkspaceFileItem }) {
  return (
    <div className="absolute inset-0">
      <Fallback file={file} />
      <ProgressBar />
    </div>
  );
}

export function ProgressBar() {
  return (
    <div
      className="absolute inset-x-0 top-0 h-0.5 overflow-hidden bg-zinc-200/70"
      aria-hidden
    >
      <div className="h-full w-1/3 animate-progress-bar rounded-full bg-zinc-500" />
    </div>
  );
}
