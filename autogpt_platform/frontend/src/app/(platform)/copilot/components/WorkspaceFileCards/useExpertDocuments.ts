"use client";

import {
  getListWorkspaceFilesQueryKey,
  useListWorkspaceFiles,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";
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

  const documents = (query.data?.files ?? [])
    .filter(
      (item) => !isInternalToolOutput(item) && !isHiddenWorkspaceFile(item),
    )
    .map<SessionFile>((item) => ({ item, messageID: null }));

  return {
    documents,
    isLoading: query.isLoading && Boolean(expertId),
    isError: query.isError,
  };
}
