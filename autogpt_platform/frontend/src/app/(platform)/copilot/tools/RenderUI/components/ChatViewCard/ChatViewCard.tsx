import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import { FloppyDiskIcon } from "@hugeicons/core-free-icons";
import type { DraftRemote } from "../../draftTargets";
import type { SavedFileRef } from "../../helpers";
import { DraftStatus } from "../DraftStatus";
import { ResponseFrame } from "../ResponseFrame";
import { SavedSummary } from "../SavedSummary";
import { SaveViewDialog } from "../SaveViewDialog/SaveViewDialog";
import { useChatViewCard } from "./useChatViewCard";

interface Props {
  ownerId: string | undefined;
  sessionId: string;
  toolCallId: string;
  viewId: string;
  stable: boolean;
  locked: boolean;
  remote: DraftRemote;
  source: string;
  title: string;
  summary: string;
  valid: boolean;
  onSaved: (file: SavedFileRef) => void;
}

export function ChatViewCard(props: Props) {
  const { source, summary, valid, locked } = props;
  const { draft, submit, canSave, isSaveOpen, setIsSaveOpen } =
    useChatViewCard(props);
  const fallback = (
    <SavedSummary
      summary={summary}
      valid={false}
      locked={locked}
      isSending={submit.isSending}
      onRebuild={() =>
        void submit.send(
          "Please rebuild the previous interactive view using the same data and valid OpenUI components.",
        )
      }
    />
  );
  return (
    <ResponseFrame
      locked={locked}
      isSending={submit.isSending}
      sent={submit.sent}
      error={submit.error}
    >
      {valid && canSave && (
        <div className="flex justify-end">
          <Button
            type="button"
            size="xs"
            variant="outline"
            leftIcon={<Icon icon={FloppyDiskIcon} size={14} />}
            onClick={() => setIsSaveOpen(true)}
          >
            Save to files
          </Button>
        </div>
      )}
      <DraftStatus draft={draft} />
      {valid && draft.ready && (
        <OpenUI
          source={source}
          isStreaming={false}
          instanceId={draft.namespace}
          disabled={locked || submit.isSending || !draft.writable}
          revision={draft.revision + 1}
          initialState={draft.initialState}
          onStateUpdate={draft.onStateUpdate}
          onAction={submit.onAction}
          fallback={fallback}
        />
      )}
      {!valid && fallback}
      {isSaveOpen && (
        <SaveViewDialog
          ownerId={props.ownerId}
          sessionId={props.sessionId}
          viewId={props.viewId}
          title={props.title}
          beforeSave={draft.flushSync}
          onClose={() => setIsSaveOpen(false)}
          onSaved={props.onSaved}
        />
      )}
    </ResponseFrame>
  );
}
