import type { ChatStatus, UIMessage } from "ai";
import { useRef } from "react";
import {
  FlatList,
  StyleSheet,
  Text,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from "react-native";

import { colors, spacing, type } from "@/lib/theme";

import { MessageBubble } from "./MessageBubble";

const PIN_THRESHOLD = 120;

interface Props {
  messages: UIMessage[];
  status: ChatStatus;
}

export function MessageList({ messages, status }: Props) {
  const listRef = useRef<FlatList<UIMessage>>(null);
  const isNearEnd = useRef(true);

  function handleScroll(event: NativeSyntheticEvent<NativeScrollEvent>) {
    const { contentOffset, contentSize, layoutMeasurement } = event.nativeEvent;
    const distance =
      contentSize.height - (contentOffset.y + layoutMeasurement.height);
    isNearEnd.current = distance < PIN_THRESHOLD;
  }

  function handleContentSizeChange() {
    if (isNearEnd.current) listRef.current?.scrollToEnd({ animated: false });
  }

  return (
    <FlatList
      ref={listRef}
      data={messages}
      keyExtractor={(message) => message.id}
      renderItem={({ item }) => <MessageBubble message={item} />}
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
});
