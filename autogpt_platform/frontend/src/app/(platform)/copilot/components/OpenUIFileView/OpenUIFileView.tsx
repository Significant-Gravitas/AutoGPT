"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import type { ReactNode } from "react";
import { FileViewEditor } from "./components/FileViewEditor/FileViewEditor";
import type { FileViewHeaderProps } from "./components/FileViewHeader";
import { StaleRevision } from "./components/StaleRevision";
import { useOpenUIFileView } from "./useOpenUIFileView";

interface Props {
  fileId: string;
  /** The chat the view is shown in, or null outside a chat. */
  currentSessionId: string | null;
  /** Shown when the file is not a saved view (or not this user's). */
  fallback: ReactNode;
  /** The revision a chat card was rendered from. A different file source
   *  means the card is an older revision, shown read-only. */
  cardSource?: string;
  header?: FileViewHeaderProps;
  readOnly?: boolean;
}

export function OpenUIFileView({
  fileId,
  currentSessionId,
  fallback,
  cardSource,
  header,
  readOnly = false,
}: Props) {
  const { view, meta, remote, isLoading, isMissing, isError, retry } =
    useOpenUIFileView(fileId);

  if (isLoading)
    return (
      <div aria-busy className="my-3 space-y-3">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  if (isMissing) return <>{fallback}</>;
  if (isError || !view || !meta)
    return (
      <div role="alert" className="my-3 space-y-2 text-sm">
        <p>This view couldn&apos;t be loaded.</p>
        <Button
          type="button"
          size="small"
          variant="secondary"
          onClick={() => void retry()}
        >
          Try again
        </Button>
      </div>
    );
  if (!meta.valid) return <>{fallback}</>;
  if (cardSource !== undefined && cardSource !== view.source)
    return <StaleRevision source={cardSource} header={header} />;
  return (
    <FileViewEditor
      key={view.checksum}
      view={view}
      remote={remote}
      title={meta.title}
      currentSessionId={currentSessionId}
      readOnly={readOnly}
      header={header}
      fallback={fallback}
    />
  );
}
