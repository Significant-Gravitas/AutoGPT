import { useMemo, useState } from "react";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

export function useArtifactsPage() {
  const [searchTerm, setSearchTerm] = useState("");

  const { data, isLoading, isError, error } = useListWorkspaceFiles(
    { limit: 1000 },
    {
      query: {
        select: (res) =>
          res.status === 200 ? (res.data.files ?? []) : ([] as WorkspaceFileItem[]),
      },
    },
  );

  const files = useMemo(() => {
    const all = data ?? [];
    const q = searchTerm.trim().toLowerCase();
    if (!q) return all;
    return all.filter((file) => file.name.toLowerCase().includes(q));
  }, [data, searchTerm]);

  return {
    files,
    isLoading,
    isError,
    error,
    searchTerm,
    setSearchTerm,
  };
}
