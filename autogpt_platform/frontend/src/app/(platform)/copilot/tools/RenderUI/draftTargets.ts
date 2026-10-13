import { putV2SaveUiDraft } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetWorkspaceFileViewStateQueryKey,
  putWorkspaceFileViewState,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { UIDraftWrite } from "@/app/api/__generated__/models/uIDraftWrite";
import type { UIFileStateWrite } from "@/app/api/__generated__/models/uIFileStateWrite";
import type { QueryKey } from "@tanstack/react-query";

export interface SavedDraft {
  source: string;
  state: Record<string, unknown>;
  revision: number;
  writable: boolean;
}

export interface DraftRemote {
  data: SavedDraft | null;
  settled: boolean;
  /** When the last load finished, so a reload can wait for fresh data. */
  version: number;
  refetch: () => Promise<unknown>;
}

// Where a view's edits are stored: the chat draft of an unsaved view, or the
// edits row of a saved `.openui` file. Merge, cache and conflict handling in
// useUIDraft/useDraftSave work the same for both.
export interface DraftTarget {
  cacheKey: string | null;
  downloadName: string;
  queryKey: QueryKey;
  write: (state: Record<string, unknown>, revision: number) => Promise<number>;
}

interface ChatDraftArgs {
  ownerId: string | undefined;
  sessionId: string;
  viewId: string;
  toolCallId: string;
}

export function chatDraftQueryKey(args: ChatDraftArgs) {
  return [
    "ui-draft",
    args.ownerId,
    args.sessionId,
    args.viewId,
    args.toolCallId,
  ] as const;
}

export function chatDraftTarget(args: ChatDraftArgs): DraftTarget {
  const { ownerId, sessionId, viewId, toolCallId } = args;
  return {
    cacheKey: ownerId
      ? `copilot-ui:v2:${ownerId}:${sessionId}:${viewId}`
      : null,
    downloadName: `${viewId}-draft.json`,
    queryKey: ["ui-draft", ownerId, sessionId, viewId],
    async write(state, revision) {
      const result = await putV2SaveUiDraft(sessionId, viewId, {
        tool_call_id: toolCallId,
        expected_revision: revision,
        state: state as UIDraftWrite["state"],
      });
      if (result.status !== 200) throw new Error("save");
      return result.data.revision;
    },
  };
}

interface FileDraftArgs {
  ownerId: string | undefined;
  fileId: string;
  checksum: string;
}

export function fileDraftTarget(args: FileDraftArgs): DraftTarget {
  const { ownerId, fileId, checksum } = args;
  return {
    cacheKey: ownerId ? `copilot-ui:v2:${ownerId}:file:${fileId}` : null,
    downloadName: `${fileId}-edits.json`,
    queryKey: getGetWorkspaceFileViewStateQueryKey(fileId),
    async write(state, revision) {
      const result = await putWorkspaceFileViewState(fileId, {
        expected_revision: revision,
        expected_checksum: checksum,
        state: state as UIFileStateWrite["state"],
      });
      if (result.status !== 200) throw new Error("save");
      return result.data.revision;
    },
  };
}
