import { useState } from "react";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

export type OriginFilter = "all" | "builder" | "autopilot";

export function useArtifactsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [originFilter, setOriginFilter] = useState<OriginFilter>("all");

  const trimmedSearch = searchTerm.trim();

  const { data, isLoading, isError, error } = useListWorkspaceFiles(
    {
      limit: 1000,
      q: trimmedSearch || undefined,
      origin: originFilter === "all" ? undefined : originFilter,
    },
    {
      query: {
        select: (res) =>
          res.status === 200
            ? (res.data.files ?? [])
            : ([] as WorkspaceFileItem[]),
        // Reduce flicker when the user types — keep the previous page of
        // results visible until the new query lands.
        placeholderData: (prev) => prev,
      },
    },
  );

  return {
    files: data ?? [],
    isLoading,
    isError,
    error,
    searchTerm,
    setSearchTerm,
    originFilter,
    setOriginFilter,
  };
}
