import { useState } from "react";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { readViewMeta, type SavedFileRef } from "./helpers";
import { useChatDraft } from "./useChatDraft";

interface Args {
  sessionId: string | null;
  toolCallId: string;
  source: string;
  file: SavedFileRef | null;
  readOnly: boolean;
}

export function useRenderUI({
  sessionId,
  toolCallId,
  source,
  file,
  readOnly,
}: Args) {
  const ownerId = useAuthStore((state) => state.user?.id);
  const { chatSurface } = useCopilotChatActions();
  const locked = readOnly || chatSurface === "share";
  const { viewId } = readViewMeta(source);
  const [savedHere, setSavedHere] = useState<SavedFileRef | null>(null);
  const known = file ?? savedHere;
  const stable = !locked && !!sessionId && !!viewId;
  const chatDraft = useChatDraft({
    ownerId,
    sessionId: sessionId ?? "",
    viewId,
    toolCallId,
    enabled: stable && !known,
  });
  const fromDraft = chatDraft.fileId
    ? { id: chatDraft.fileId, name: null }
    : null;

  return {
    ownerId,
    locked,
    viewId,
    stable,
    remote: chatDraft.remote,
    savedFile: locked ? null : (known ?? fromDraft),
    onSaved: setSavedHere,
  };
}
