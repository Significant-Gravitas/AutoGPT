"use client";

import React, { useState } from "react";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Button } from "@/components/atoms/Button/Button";
import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import { formatMaybeJson } from "../../helpers";

interface Props {
  output: BlockOutputResponse;
}

const COLLAPSED_LIMIT = 3;

function resolveWorkspaceUrl(src: string): string {
  const withoutPrefix = src.replace("workspace://", "");
  const fileId = withoutPrefix.split("#")[0];
  const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
  return `/api/proxy${apiPath}`;
}

function getWorkspaceMimeHint(src: string): string | undefined {
  const hashIndex = src.indexOf("#");
  if (hashIndex === -1) return undefined;
  return src.slice(hashIndex + 1) || undefined;
}

function isWorkspaceRef(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

function WorkspaceMedia({ value }: { value: string }) {
  const [imgFailed, setImgFailed] = useState(false);
  const resolvedUrl = resolveWorkspaceUrl(value);
  const mime = getWorkspaceMimeHint(value);

  if (mime?.startsWith("video/") || imgFailed) {
    return (
      <video
        controls
        className="mt-2 h-auto max-w-full rounded-md border border-zinc-200"
        preload="metadata"
      >
        <source src={resolvedUrl} />
      </video>
    );
  }

  if (mime?.startsWith("audio/")) {
    return <audio controls src={resolvedUrl} className="mt-2 w-full" />;
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={resolvedUrl}
      alt="Output media"
      className="mt-2 h-auto max-w-full rounded-md border border-zinc-200"
      loading="lazy"
      onError={() => setImgFailed(true)}
    />
  );
}

function renderOutputValue(value: unknown): React.ReactNode {
  if (isWorkspaceRef(value)) {
    return <WorkspaceMedia value={value} />;
  }
  if (Array.isArray(value)) {
    const hasWorkspace = value.some(isWorkspaceRef);
    if (hasWorkspace) {
      return (
        <>
          {value.map((item, i) =>
            isWorkspaceRef(item) ? (
              <WorkspaceMedia key={i} value={item} />
            ) : (
              <pre
                key={i}
                className="mt-1 whitespace-pre-wrap text-xs text-muted-foreground"
              >
                {formatMaybeJson(item)}
              </pre>
            ),
          )}
        </>
      );
    }
  }
  return null;
}

function OutputKeySection({
  outputKey,
  items,
}: {
  outputKey: string;
  items: unknown[];
}) {
  const [expanded, setExpanded] = useState(false);
  const mediaContent = renderOutputValue(items);
  const hasMoreItems = items.length > COLLAPSED_LIMIT;
  const visibleItems = expanded ? items : items.slice(0, COLLAPSED_LIMIT);

  return (
    <div className="rounded-2xl border bg-background p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-xs font-medium text-foreground">
          {outputKey}
        </p>
        <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
          {items.length} item{items.length === 1 ? "" : "s"}
        </span>
      </div>
      {mediaContent || (
        <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
          {formatMaybeJson(visibleItems)}
        </pre>
      )}
      {!mediaContent && hasMoreItems && (
        <Button
          variant="ghost"
          size="small"
          className="mt-1 h-auto px-0 py-0.5 text-[11px] text-muted-foreground"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Show less" : `Show all ${items.length} items`}
        </Button>
      )}
    </div>
  );
}

export function BlockOutputCard({ output }: Props) {
  return (
    <div className="grid gap-2">
      <p className="text-sm text-foreground">{output.message}</p>

      {Object.entries(output.outputs ?? {}).map(([key, items]) => (
        <OutputKeySection key={key} outputKey={key} items={items} />
      ))}
    </div>
  );
}
