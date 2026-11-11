import { skipToken, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  listExpertFiles,
  listSessionFiles,
  workspaceKeys,
  type WorkspaceFile,
} from "@/lib/api/workspace";

import { visibleFiles } from "./helpers";

interface Args {
  sessionID: string;
  expertID: string | null;
  isOpen: boolean;
}

export function useFilesPanel({ sessionID, expertID, isOpen }: Args) {
  const [openFile, setOpenFile] = useState<WorkspaceFile | null>(null);
  // Fetched only while the panel is open; each opening refreshes stale lists,
  // so files Otto wrote since the last look show up.
  const sessionFiles = useQuery({
    queryKey: workspaceKeys.sessionFiles(sessionID),
    queryFn: () => listSessionFiles(sessionID),
    enabled: isOpen,
    select: visibleFiles,
  });
  const expertFiles = useQuery({
    queryKey: workspaceKeys.expertFiles(expertID ?? ""),
    queryFn: expertID ? () => listExpertFiles(expertID) : skipToken,
    enabled: isOpen,
    select: visibleFiles,
  });

  function closeFile() {
    setOpenFile(null);
  }

  return {
    session: toSection(sessionFiles),
    expert: expertID ? toSection(expertFiles) : null,
    openFile,
    selectFile: setOpenFile,
    closeFile,
  };
}

interface FilesQuery {
  data: WorkspaceFile[] | undefined;
  isPending: boolean;
  isError: boolean;
  refetch: () => Promise<unknown>;
}

function toSection(query: FilesQuery) {
  return {
    files: query.data ?? [],
    // Data first: a failed background refetch keeps showing the last list.
    isLoading: !query.data && query.isPending,
    isError: !query.data && query.isError,
    retry() {
      void query.refetch();
    },
  };
}
