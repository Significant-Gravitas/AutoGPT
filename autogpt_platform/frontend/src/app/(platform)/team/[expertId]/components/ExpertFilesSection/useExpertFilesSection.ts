"use client";

import { useState } from "react";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";
import { isInternalToolOutput } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";

export type ExpertFileFilter = "all" | "generated" | "uploaded";

const FILES_LIMIT = 200;

export function useExpertFilesSection(expertId: string) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ExpertFileFilter>("all");
  const filesQuery = useListWorkspaceFiles(
    { expert_id: expertId, limit: FILES_LIMIT },
    { query: { select: okData } },
  );

  const files = (filesQuery.data?.files ?? []).filter(
    (file) => !isInternalToolOutput(file) && !isHiddenWorkspaceFile(file),
  );
  const needle = query.trim().toLowerCase();
  const visible = files.filter(
    (file) =>
      (filter === "all" || file.origin === filter) &&
      (!needle || file.name.toLowerCase().includes(needle)),
  );

  return {
    query,
    setQuery,
    filter,
    setFilter,
    files,
    visible,
    hasMore: Boolean(filesQuery.data?.has_more),
    isLoading: filesQuery.isLoading,
    isError: filesQuery.isError,
    refetch: filesQuery.refetch,
  };
}
