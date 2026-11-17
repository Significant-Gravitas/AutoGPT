import { FlashList, type FlashListRef } from "@shopify/flash-list";
import type { ChatStatus, UIMessage } from "ai";
import type { RefObject } from "react";
import { ActivityIndicator, Pressable, Text, type LayoutChangeEvent } from "react-native";
import type { SharedValue } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { groupTurns, type Turn } from "../turns";
import { BOTTOM_PADDING } from "../useMessageList";
import { ChatScrollContext, ChatScrollView } from "./ChatScrollView";
import { MessageTurn } from "./MessageTurn";

interface Props {
  sessionID: string;
  messages: UIMessage[];
  status: ChatStatus;
  hasEarlier: boolean;
  isLoadingEarlier: boolean;
  onLoadEarlier: () => void;
  onSend: (text: string) => boolean;
  listRef: RefObject<FlashListRef<Turn> | null>;
  blankSpace: SharedValue<number>;
  composerHeight: SharedValue<number>;
  isAnchored: boolean;
  onEndVisible: (isVisible: boolean) => void;
  onViewportLayout: (event: LayoutChangeEvent) => void;
  onLastTurnLayout: (index: number, height: number) => void;
  onLoad: () => void;
}

export function MessageList({
  sessionID,
  messages,
  status,
  hasEarlier,
  isLoadingEarlier,
  onLoadEarlier,
  onSend,
  listRef,
  blankSpace,
  composerHeight,
  isAnchored,
  onEndVisible,
  onViewportLayout,
  onLastTurnLayout,
  onLoad,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const isBusy = status === "submitted" || status === "streaming";
  const turns = groupTurns(messages);
  const lastIndex = turns.length - 1;

  function renderHeader() {
    if (isLoadingEarlier) {
      return <ActivityIndicator size="small" color={colors.textTertiary} />;
    }
    if (!hasEarlier) return null;
    return (
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ disabled: isBusy }}
        disabled={isBusy}
        onPress={onLoadEarlier}
        style={({ pressed }) => [
          styles.earlier,
          pressed && styles.pressed,
          isBusy && styles.disabled,
        ]}
      >
        <Text style={styles.earlierLabel}>Load earlier messages</Text>
      </Pressable>
    );
  }

  function handleLastTurnLayout(event: LayoutChangeEvent) {
    onLastTurnLayout(lastIndex, event.nativeEvent.layout.height);
  }

  return (
    <ChatScrollContext
      value={{
        blankSpace,
        extraContentPadding: composerHeight,
        // The composer's safe-area padding sits over the keyboard once it opens.
        offset: insets.bottom,
        onEndVisible,
      }}
    >
      <FlashList
        ref={listRef}
        data={turns}
        keyExtractor={(turn) => turn.key}
        renderItem={({ item, index }) => (
          <MessageTurn
            turn={item}
            sessionID={sessionID}
            isLast={index === lastIndex}
            status={status}
            onSend={onSend}
            onLayout={index === lastIndex ? handleLastTurnLayout : undefined}
          />
        )}
        extraData={status}
        renderScrollComponent={ChatScrollView}
        initialScrollIndex={lastIndex > 0 ? lastIndex : undefined}
        onLoad={onLoad}
        onLayout={onViewportLayout}
        // Before the first send the list keeps to its end as replies load;
        // after it, the sent message holds the top and nothing chases the stream.
        maintainVisibleContentPosition={{
          autoscrollToBottomThreshold: isAnchored ? undefined : 0.15,
          animateAutoScrollToBottom: false,
        }}
        ListHeaderComponent={renderHeader()}
        style={styles.list}
        contentContainerStyle={styles.content}
      />
    </ChatScrollContext>
  );
}

const useStyles = createStyles((colors) => ({
  list: { flex: 1 },
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: BOTTOM_PADDING,
  },
  earlier: {
    alignSelf: "center",
    minHeight: 36,
    justifyContent: "center",
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  disabled: { opacity: 0.4 },
  earlierLabel: { ...type.small, color: colors.textSecondary },
}));
