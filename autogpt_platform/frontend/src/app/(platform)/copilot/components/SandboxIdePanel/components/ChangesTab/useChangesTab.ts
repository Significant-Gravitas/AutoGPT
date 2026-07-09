import {
  useGetV2GetSandboxChanges,
  useGetV2GetSandboxDiff,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SandboxChangesResponse } from "@/app/api/__generated__/models/sandboxChangesResponse";
import type { SandboxDiffResponse } from "@/app/api/__generated__/models/sandboxDiffResponse";
import { useState } from "react";

export function useChangesTab(sessionId: string) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const {
    data: changes,
    isLoading,
    isError,
  } = useGetV2GetSandboxChanges(sessionId, {
    query: { select: (res) => res.data as SandboxChangesResponse },
  });

  const { data: diff, isLoading: isDiffLoading } = useGetV2GetSandboxDiff(
    sessionId,
    { path: selectedPath ?? "" },
    {
      query: {
        enabled: !!selectedPath,
        select: (res) => res.data as SandboxDiffResponse,
      },
    },
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
