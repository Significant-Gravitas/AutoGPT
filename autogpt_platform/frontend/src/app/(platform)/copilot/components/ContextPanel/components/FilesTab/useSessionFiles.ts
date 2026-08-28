"use client";

import {
  getListWorkspaceFilesQueryKey,
  useListWorkspaceFiles,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useCopilotStreamStore } from "../../../../copilotStreamStore";
import { useCopilotUIStore } from "../../../../store";
import { getMessageArtifacts } from "../../../ChatMessagesContainer/helpers";
import { isUploadedFile } from "./helpers";

export interface SessionFile {
  item: WorkspaceFileItem;
  messageID: string | null;
}

export function useSessionFiles(sessionId: string | null) {
  const tenantScope = useCopilotUIStore((s) => s.artifactTenantScope);
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );

  const query = useListWorkspaceFiles(
    { session_id: sessionId ?? undefined },
    {
      query: {
        enabled: !!sessionId && tenantScope !== null,
        queryKey: getTeamScopedQueryKey(
          getListWorkspaceFilesQueryKey({
            session_id: sessionId ?? undefined,
          }),
          tenantScope?.organizationId,
          tenantScope?.teamId,
        ),
        select: (res) => res.data as ListFilesResponse,
      },
      request: getTenantRequestInit(
        tenantScope?.organizationId,
        tenantScope?.teamId,
        tenantScope !== null,
      ),
    },
  );

  const fileIdToMessageId = new Map<string, string>();
  for (const message of messages ?? []) {
    for (const artifact of getMessageArtifacts(message)) {
      if (!fileIdToMessageId.has(artifact.id)) {
        fileIdToMessageId.set(artifact.id, message.id);
      }
    }
  }

  const files: SessionFile[] = (query.data?.files ?? []).map((item) => ({
    item,
    messageID: fileIdToMessageId.get(item.id) ?? null,
  }));

  const uploaded = files.filter((f) => isUploadedFile(f.item));
  const generated = files.filter((f) => !isUploadedFile(f.item));

  return {
    uploaded,
    generated,
    isLoading: query.isLoading && !!sessionId && tenantScope !== null,
    isError: query.isError,
    error: query.error,
    isEmpty: !!sessionId && tenantScope !== null && files.length === 0,
  };
}
