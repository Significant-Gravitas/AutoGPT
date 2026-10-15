"use client";

import { useGetExpertDetachPreview } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";
import { isInternalToolOutput } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";
import { getPauseItems } from "../../components/FireExpertDialog/helpers";

const FILES_LIMIT = 200;

/** What firing would touch, read ahead of time so the card can say so. */
export function useExpertSettingsSection(expertId: string) {
  const previewQuery = useGetExpertDetachPreview(expertId, {
    query: { select: (res) => okData(res) ?? null },
  });
  const filesQuery = useListWorkspaceFiles(
    { expert_id: expertId, limit: FILES_LIMIT },
    { query: { select: okData } },
  );

  const files = (filesQuery.data?.files ?? []).filter(
    (file) => !isInternalToolOutput(file) && !isHiddenWorkspaceFile(file),
  );

  return {
    pauseItems: getPauseItems(previewQuery.data ?? null),
    isPreviewLoading: previewQuery.isLoading,
    isPreviewError: previewQuery.isError,
    retryPreview: () => previewQuery.refetch(),
    fileCount: files.length,
    hasMoreFiles: Boolean(filesQuery.data?.has_more),
  };
}
