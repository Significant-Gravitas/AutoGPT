"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { WatcherMetadata } from "./helpers";

interface Props {
  metadata: WatcherMetadata;
}

export function WatcherCard({ metadata }: Props) {
  return (
    <div
      className="max-w-md rounded-2xl bg-white p-4 ring-1 ring-inset ring-zinc-200"
      data-testid="watcher-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-zinc-900">{metadata.title}</p>
          <p className="mt-1 text-sm text-zinc-500">{metadata.description}</p>
        </div>
        <span
          className={
            metadata.status === "failed"
              ? "rounded-full bg-red-50 px-2 py-1 text-xs font-medium text-red-700"
              : "rounded-full bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700"
          }
        >
          {metadata.status === "failed" ? "Failed" : "Needs you"}
        </span>
      </div>
      <Button
        as="NextLink"
        href={metadata.actionHref}
        variant="secondary"
        size="small"
        className="mt-3"
      >
        {metadata.actionLabel}
      </Button>
    </div>
  );
}
