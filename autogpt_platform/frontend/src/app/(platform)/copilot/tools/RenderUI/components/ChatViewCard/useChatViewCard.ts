import { useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { chatDraftTarget, type DraftRemote } from "../../draftTargets";
import { useUIDraft } from "../../useUIDraft";
import { useViewSubmit } from "../../useViewSubmit";

interface Args {
  ownerId: string | undefined;
  sessionId: string;
  toolCallId: string;
  viewId: string;
  stable: boolean;
  locked: boolean;
  remote: DraftRemote;
  source: string;
  title: string;
}

export function useChatViewCard(args: Args) {
  const { ownerId, sessionId, toolCallId, viewId, stable, locked } = args;
  const { onSend } = useCopilotChatActions();
  const [isSaveOpen, setIsSaveOpen] = useState(false);
  const draft = useUIDraft({
    namespace: `${sessionId}:${viewId}`,
    target: stable
      ? chatDraftTarget({ ownerId, sessionId, viewId, toolCallId })
      : null,
    remote: stable ? args.remote : null,
    source: args.source,
    locked,
    legacyKey: sessionId ? `copilot-ui:${sessionId}:${toolCallId}` : null,
  });
  const submit = useViewSubmit({
    title: args.title,
    disabled: locked || !draft.writable,
    deliver: onSend,
  });

  return {
    draft,
    submit,
    canSave: stable && draft.ready && draft.writable,
    isSaveOpen,
    setIsSaveOpen,
  };
}
