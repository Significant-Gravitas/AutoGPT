import { KeyboardAvoidingView, Platform, StyleSheet, Text } from "react-native";

import type { SessionDetail } from "@/lib/api/chat";
import { colors, spacing, type } from "@/lib/theme";

import { useChatThread } from "../useChatThread";
import { Composer } from "./Composer";
import { MessageList } from "./MessageList";

interface Props {
  sessionID: string;
  detail: SessionDetail;
}

export function ChatThread({ sessionID, detail }: Props) {
  const { messages, status, error, isBusy, send, stop } = useChatThread({
    sessionID,
    detail,
  });

  function handleStop() {
    void stop();
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.thread}
    >
      <MessageList messages={messages} status={status} />
      {error ? <Text style={styles.error}>{error.message}</Text> : null}
      <Composer onSend={send} onStop={handleStop} isBusy={isBusy} />
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  thread: { flex: 1 },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
});
