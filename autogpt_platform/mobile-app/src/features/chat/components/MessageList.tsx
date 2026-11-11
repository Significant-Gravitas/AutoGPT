import type { ChatStatus, UIMessage } from "ai";
import { useRef } from "react";
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from "react-native";

import { colors, radius, spacing, type } from "@/lib/theme";

import { MessageBubble } from "./MessageBubble";

const PIN_THRESHOLD = 120;

interface Props {
  messages: UIMessage[];
  status: ChatStatus;
  hasEarlier: boolean;
  isLoadingEarlier: boolean;
  onLoadEarlier: () => void;
}

export function MessageList({
  messages,
  status,
  hasEarlier,
  isLoadingEarlier,
  onLoadEarlier,
}: Props) {
  const listRef = useRef<FlatList<UIMessage>>(null);
  const isNearEnd = useRef(true);
  const isBusy = status === "submitted" || status === "streaming";

  function handleScroll(event: NativeSyntheticEvent<NativeScrollEvent>) {
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;
    const distance =
      contentSize.height - (contentOffset.y + layoutMeasurement.height);
    isNearEnd.current = distance < PIN_THRESHOLD;
  }

  function handleContentSizeChange() {
    if (isNearEnd.current) listRef.current?.scrollToEnd({ animated: false });
  }

  function renderHeader() {
    if (isLoadingEarlier) {
      return <ActivityIndicator size="small" color={colors.textTertiary} />;
    }
    if (!hasEarlier) return undefined;
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

  return (
    <FlatList
      ref={listRef}
      data={messages}
      keyExtractor={(message) => message.id}
      renderItem={({ item }) => <MessageBubble message={item} />}
      ListHeaderComponent={renderHeader()}
      maintainVisibleContentPosition={{ minIndexForVisible: 0 }}
      ListFooterComponent={
        status === "submitted" ? (
          <Text style={styles.thinking}>Thinking…</Text>
        ) : undefined
      }
      onScroll={handleScroll}
      onContentSizeChange={handleContentSizeChange}
      scrollEventThrottle={32}
      keyboardShouldPersistTaps="handled"
      style={styles.list}
      contentContainerStyle={styles.content}
    />
  );
}

const styles = StyleSheet.create({
  list: { flex: 1 },
  content: {
    gap: spacing.lg,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.lg,
  },
  thinking: { ...type.small, color: colors.textTertiary },
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
});
