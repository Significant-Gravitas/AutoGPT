import type { UIDataTypes, UIMessage, UITools } from "ai";
import { MessageActions } from "@/components/organisms/Message/Message";
import { AssistantMessageActions } from "../../AssistantMessageActions";
import { CopyButton } from "../../CopyButton";

interface Props {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  isLastAssistant: boolean;
  readOnly: boolean;
  copyText: string;
  sessionID?: string | null;
  savedMessageId?: string | null;
}

export function AssistantRowActions({
  message,
  isLastAssistant,
  readOnly,
  copyText,
  sessionID,
  savedMessageId,
}: Props) {
  if (!readOnly) {
    return (
      <AssistantMessageActions
        message={message}
        sessionID={sessionID ?? null}
        savedMessageId={savedMessageId}
        reveal={isLastAssistant ? undefined : "hover"}
      />
    );
  }
  return (
    <MessageActions
      reveal={isLastAssistant ? undefined : "hover"}
      className="mt-1 items-center justify-start gap-2"
    >
      <CopyButton text={copyText} />
    </MessageActions>
  );
}
