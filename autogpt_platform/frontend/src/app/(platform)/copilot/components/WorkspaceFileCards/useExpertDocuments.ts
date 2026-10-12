"use client";

import {
  getListWorkspaceFilesQueryKey,
  useListWorkspaceFiles,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { useMemo } from "react";
import { isInternalToolOutput } from "../ContextPanel/components/FilesTab/helpers";
import type { SessionFile } from "../ContextPanel/components/FilesTab/useSessionFiles";

export function useExpertDocuments(
  expertId: string | null,
  sessionDocumentCount: number,
) {
  const params = {
    expert_id: expertId ?? undefined,
    origin: "generated" as const,
    limit: 200,
  };
  const query = useListWorkspaceFiles(params, {
    query: {
      enabled: Boolean(expertId),
      queryKey: [
        ...getListWorkspaceFilesQueryKey(params),
        sessionDocumentCount,
      ],
      placeholderData: (previous) => previous,
      select: (res) => res.data as ListFilesResponse,
    },
  });

  const documents = useMemo(
    () =>
      (query.data?.files ?? [])
        .filter((item) => !isInternalToolOutput(item))
        .map<SessionFile>((item) => ({ item, messageID: null })),
    [query.data],
  );

  return {
    documents,
    isLoading: query.isLoading && Boolean(expertId),
    isError: query.isError,
  };
}
