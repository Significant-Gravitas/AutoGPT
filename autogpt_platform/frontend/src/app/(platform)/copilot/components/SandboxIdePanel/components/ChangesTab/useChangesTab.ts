import {
  useGetV2GetSandboxChanges,
  useGetV2GetSandboxDiff,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useState } from "react";

export function useChangesTab(sessionId: string) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const {
    data: changes,
    isLoading,
    isError,
  } = useGetV2GetSandboxChanges(sessionId, {
    query: { select: (res) => res.data },
  });

  const { data: diff, isLoading: isDiffLoading } = useGetV2GetSandboxDiff(
    sessionId,
    { path: selectedPath ?? "" },
    { query: { enabled: !!selectedPath, select: (res) => res.data } },
  );

  return {
    changes,
    isLoading,
    isError,
    selectedPath,
    setSelectedPath,
    diff,
    isDiffLoading,
  };
}
