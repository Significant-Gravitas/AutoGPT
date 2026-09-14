"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Skeleton } from "@/components/ui/skeleton";
import { useCopilotUIStore } from "../../../../store";
import { downloadArtifact } from "../../../ArtifactPanel/downloadArtifact";
import { fileItemToArtifactRef } from "../FilesTab/helpers";
import type { SessionFile } from "../FilesTab/useSessionFiles";
import { useSessionFiles } from "../FilesTab/useSessionFiles";
import { ArtifactMiniCard } from "./components/ArtifactMiniCard";

// The panel previews what this chat produced; /artifacts stays one click away
// for the workspace-wide library.
const PREVIEWABLE_LIMIT = 6;

interface Props {
  sessionId: string | null;
}

export function ArtifactsTab({ sessionId }: Props) {
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const { uploaded, generated, isLoading, isError } =
    useSessionFiles(sessionId);
  const files = [...generated, ...uploaded].slice(0, PREVIEWABLE_LIMIT);

  function handleDownload(file: SessionFile) {
    downloadArtifact(fileItemToArtifactRef(file.item)).catch(() =>
      toast({ title: "Download failed", variant: "destructive" }),
    );
  }

  if (isLoading) {
    return (
      <div className="mx-auto flex w-full max-w-[17rem] flex-col gap-2 p-3">
        <Skeleton className="h-12 w-full rounded-2xl" />
        <Skeleton className="h-12 w-full rounded-2xl" />
        <Skeleton className="h-12 w-full rounded-2xl" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-3">
        <ErrorCard
          isSuccess={false}
          context="artifacts"
          responseError={{ message: "Failed to load artifacts." }}
        />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-4 py-8">
      <div className="flex flex-col items-center gap-3">
        <p className="max-w-[17rem] text-center text-sm text-zinc-500">
          {files.length === 0
            ? "Nothing to preview yet."
            : "Pick an artifact from this chat to preview it here."}
        </p>
        <Button
          as="NextLink"
          href="/artifacts"
          variant="secondary"
          size="small"
        >
          Open artifacts
        </Button>
      </div>
      {files.length > 0 && (
        <div className="flex w-full max-w-[17rem] flex-col gap-2">
          {files.map((file) => (
            <ArtifactMiniCard
              key={file.item.id}
              file={file}
              onOpen={() => openArtifact(fileItemToArtifactRef(file.item))}
              onDownload={handleDownload}
            />
          ))}
        </div>
      )}
    </div>
  );
}
