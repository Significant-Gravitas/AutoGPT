"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useCallback, useState } from "react";
import { getPreviewKind, type PreviewKind } from "../helpers";
import { ImagePreview, VideoPreview } from "./MediaPreview";
import { Fallback } from "./PreviewParts";
import { IcsPreview, VcardPreview } from "./StructuredPreviews";
import { CsvPreview, MarkdownPreview } from "./TextPreviews";

interface Props {
  file: WorkspaceFileItem;
}

export function CardPreview({ file }: Props) {
  const kind = getPreviewKind(file.mime_type, file.size_bytes, file.name);

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

  if (hasError || kind === "none") return <Fallback file={file} />;

  switch (kind) {
    case "image":
    case "pdf":
    case "office":
      return <ImagePreview file={file} onError={handleError} />;
    case "video":
      return <VideoPreview file={file} onError={handleError} />;
    case "csv":
      return <CsvPreview file={file} onError={handleError} />;
    case "markdown":
      return <MarkdownPreview file={file} onError={handleError} />;
    case "ics":
      return <IcsPreview file={file} onError={handleError} />;
    case "vcard":
      return <VcardPreview file={file} onError={handleError} />;
    // json / text / code render as a generic file card (with a language
    // badge) — the grid is for scanning, not reading raw source.
    default:
      return <Fallback file={file} />;
  }
}
