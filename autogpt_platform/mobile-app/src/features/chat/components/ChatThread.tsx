import { Text, View } from "react-native";
import { KeyboardStickyView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import type { SessionDetail } from "@/lib/api/chat";
import { tapSend, tapStop } from "@/lib/haptics";
import { createStyles, spacing, type } from "@/lib/theme";

import { groupTurns } from "../turns";
import { useAttachments } from "../useAttachments";
import { useChatThread } from "../useChatThread";
import { useMessageList } from "../useMessageList";
import { Composer } from "./Composer";
import { MessageList } from "./MessageList";
import { ScrollToEndButton } from "./ScrollToEndButton";

interface Props {
  sessionID: string;
  detail: SessionDetail;
  /** The detail came from the cache, so the thread checks the server on mount. */
  isCached: boolean;
}

export function ChatThread({ sessionID, detail, isCached }: Props) {
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
  } = useChatThread({ sessionID, detail, isCached });
  const files = useAttachments(sessionID);
  const list = useMessageList();
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const line = files.message ?? errorMessage;

  function sendAndAnchor(text: string, withFiles: boolean) {
    const turnCount = groupTurns(messages).length;
    const isSent = withFiles ? send(text, files.readyFiles) : send(text);
    if (isSent) list.anchorNextTurn(turnCount);
    return isSent;
  }

  function handleSend(text: string) {
    const isSent = sendAndAnchor(text, true);
    if (isSent) files.clear();
    return isSent;
  }

  // Card answers go out on their own, without the composer's attachments.
  function handleAnswer(text: string) {
    const isSent = sendAndAnchor(text, false);
    if (isSent) tapSend();
    return isSent;
  }

  function handleAttach() {
    void files.attach();
  }

  function handleStop() {
    tapStop();
    void stop();
  }

  function handleLoadEarlier() {
    void loadEarlier().catch(() => undefined);
  }

  return (
    <View style={styles.thread}>
      <MessageList
        sessionID={sessionID}
        messages={messages}
        status={status}
        hasEarlier={hasEarlier}
        isLoadingEarlier={isLoadingEarlier}
        onLoadEarlier={handleLoadEarlier}
        onSend={handleAnswer}
        listRef={list.listRef}
        blankSpace={list.blankSpace}
        composerHeight={list.composerHeight}
        isAnchored={list.isAnchored}
        onEndVisible={list.setIsAtEnd}
        onViewportLayout={list.handleViewportLayout}
        onLastTurnLayout={list.handleLastTurnLayout}
        onLoad={list.jumpToEnd}
      />
      <KeyboardStickyView
        // The composer's safe-area padding tucks under the keyboard once it opens.
        offset={{ closed: 0, opened: insets.bottom }}
        style={styles.dock}
      >
        <View onLayout={list.handleComposerLayout}>
          {!list.isAtEnd && messages.length > 0 ? (
            <ScrollToEndButton onPress={list.scrollToEnd} />
          ) : null}
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
        </View>
      </KeyboardStickyView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  thread: { flex: 1 },
  dock: { position: "absolute", left: 0, right: 0, bottom: 0 },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
  },
}));
