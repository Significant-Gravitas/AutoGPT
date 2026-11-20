"use client";

import {
  getListWorkspaceFilesQueryKey,
  useListWorkspaceFiles,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";
import { useMemo } from "react";
import { useCopilotStreamStore } from "../../../../copilotStreamStore";
import { getMessageArtifacts } from "../../../ChatMessagesContainer/helpers";
import { isInternalToolOutput, isUploadedFile } from "./helpers";

export interface SessionFile {
  item: WorkspaceFileItem;
  messageID: string | null;
}

export function useSessionFiles(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  // Kept as a memo: the map is read again after the query hook below, and the
  // compiler cannot cache a value across a hook call, so without this it would
  // be rebuilt, and the file lists with it, on every render.
  const { documentCount, fileIdToMessageId } = useMemo(() => {
    const documentIds = new Set<string>();
    const fileIdToMessageId = new Map<string, string>();
    for (const message of messages ?? []) {
      for (const artifact of getMessageArtifacts(message)) {
        if (!fileIdToMessageId.has(artifact.id)) {
          fileIdToMessageId.set(artifact.id, message.id);
        }
        if (
          message.role === "assistant" &&
          !isHiddenWorkspaceFile({ name: artifact.title })
        ) {
          documentIds.add(artifact.id);
        }
      }
    }
    return { documentCount: documentIds.size, fileIdToMessageId };
  }, [messages]);
  const params = { session_id: sessionId ?? undefined };

  const query = useListWorkspaceFiles(params, {
    query: {
      enabled: !!sessionId,
      queryKey: [...getListWorkspaceFilesQueryKey(params), documentCount],
      placeholderData: (previous) => previous,
      select: (res) => res.data as ListFilesResponse,
    },
  });

  // The chip and the artifacts button both read this on every render; the
  // compiler caches the file lists, so they keep their identity until the
  // files change.
  const files: SessionFile[] = (query.data?.files ?? [])
    .filter((item) => !isHiddenWorkspaceFile(item))
    .map((item) => ({
      item,
      messageID: fileIdToMessageId.get(item.id) ?? null,
    }));
  const uploaded = files.filter((f) => isUploadedFile(f.item));
  const generated = files.filter((f) => !isUploadedFile(f.item));
  const deliverables = files.filter(
    (f) => !isUploadedFile(f.item) && !isInternalToolOutput(f.item),
  );

  return {
    documentCount,
    uploaded,
    generated,
    deliverables,
    isLoading: query.isLoading && !!sessionId,
    isError: query.isError,
    error: query.error,
    isEmpty: !!sessionId && files.length === 0,
  };
}
