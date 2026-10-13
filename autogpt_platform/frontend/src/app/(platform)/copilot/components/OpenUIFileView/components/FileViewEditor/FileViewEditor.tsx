import type { UIFileStateResponse } from "@/app/api/__generated__/models/uIFileStateResponse";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import type { ReactNode } from "react";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { DraftRemote } from "../../../../tools/RenderUI/draftTargets";
import { DraftStatus } from "../../../../tools/RenderUI/components/DraftStatus";
import { ResponseFrame } from "../../../../tools/RenderUI/components/ResponseFrame";
import { FileViewHeader, type FileViewHeaderProps } from "../FileViewHeader";
import { useFileViewEditor } from "./useFileViewEditor";

interface Props {
  view: UIFileStateResponse;
  remote: DraftRemote;
  title: string;
  currentSessionId: string | null;
  readOnly: boolean;
  header?: FileViewHeaderProps;
  fallback: ReactNode;
}

export function FileViewEditor({ header, fallback, ...args }: Props) {
  const { draft, submit, locked, surface, actionsDisabled, note, syncStatus } =
    useFileViewEditor(args);
  return (
    <ResponseFrame
      locked={locked}
      isSending={submit.isSending}
      sent={submit.sent}
      error={submit.error}
      note={note}
    >
      {header && (
        <FileViewHeader {...header} status={locked ? undefined : syncStatus} />
      )}
      <DraftStatus draft={draft} hideSaving={!!header} />
      {draft.ready && (
        <CopilotChatActionsProvider
          onSend={(message) => submit.send(message)}
          chatSurface={surface}
        >
          <OpenUI
            source={args.view.source}
            isStreaming={false}
            instanceId={draft.namespace}
            disabled={locked || submit.isSending || !draft.writable}
            actionsDisabled={actionsDisabled}
            revision={draft.revision + 1}
            initialState={draft.initialState}
            onStateUpdate={draft.onStateUpdate}
            onAction={submit.onAction}
            fallback={fallback}
          />
        </CopilotChatActionsProvider>
      )}
    </ResponseFrame>
  );
}
