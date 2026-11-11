import { KeyboardAvoidingView, Platform, Text } from "react-native";

import type { SessionDetail } from "@/lib/api/chat";
import { createStyles, spacing, type } from "@/lib/theme";

import { useAttachments } from "../useAttachments";
import { useChatThread } from "../useChatThread";
import { Composer } from "./Composer";
import { MessageList } from "./MessageList";

interface Props {
  sessionID: string;
  detail: SessionDetail;
}

export function ChatThread({ sessionID, detail }: Props) {
  const {
    messages,
    status,
    errorMessage,
    isBusy,
    send,
    stop,
    hasEarlier,
    isLoadingEarlier,
    loadEarlier,
  } = useChatThread({ sessionID, detail });
  const files = useAttachments(sessionID);
  const styles = useStyles();
  const line = files.message ?? errorMessage;

  function handleSend(text: string) {
    const isSent = send(text, files.readyFiles);
    if (isSent) files.clear();
    return isSent;
  }

  // Card answers go out on their own, without the composer's attachments.
  function handleAnswer(text: string) {
    return send(text);
  }

  function handleAttach() {
    void files.attach();
  }

  function handleStop() {
    void stop();
  }

  function handleLoadEarlier() {
    void loadEarlier().catch(() => undefined);
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.thread}
    >
      <MessageList
        messages={messages}
        status={status}
        hasEarlier={hasEarlier}
        isLoadingEarlier={isLoadingEarlier}
        onLoadEarlier={handleLoadEarlier}
        onSend={handleAnswer}
      />
      {line ? <Text style={styles.error}>{line}</Text> : null}
      <Composer
        onSend={handleSend}
        onStop={handleStop}
        isBusy={isBusy}
        onAttach={handleAttach}
        attachments={files.attachments}
        onRemoveAttachment={files.remove}
        onRetryAttachment={files.retry}
      />
    </KeyboardAvoidingView>
  );
}

const useStyles = createStyles((colors) => ({
  thread: { flex: 1 },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
}));
