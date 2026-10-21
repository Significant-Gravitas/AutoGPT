import type { UIDataTypes, UIMessage, UITools } from "ai";
import { MessageActions } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import { AssistantMessageActions } from "../../AssistantMessageActions";
import { CopyButton } from "../../CopyButton";

interface Props {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  isLastAssistant: boolean;
  readOnly: boolean;
  copyText: string;
  sessionID?: string | null;
}

const HOVER_REVEAL =
  "opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100";

export function AssistantRowActions({
  message,
  isLastAssistant,
  readOnly,
  copyText,
  sessionID,
}: Props) {
  if (!readOnly) {
    return (
      <AssistantMessageActions
        message={message}
        sessionID={sessionID ?? null}
        className={cn(!isLastAssistant && HOVER_REVEAL)}
      />
    );
  }
  return (
    <MessageActions
      className={cn(
        "mt-1 items-center justify-start gap-2",
        !isLastAssistant && HOVER_REVEAL,
      )}
    >
      <CopyButton text={copyText} />
    </MessageActions>
  );
}
