import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useLocalSearchParams } from "expo-router";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { ChatThread } from "@/features/chat/components/ChatThread";
import { useChatScreen } from "@/features/chat/useChatScreen";
import { colors, radius, spacing, type } from "@/lib/theme";

export default function ChatScreen() {
  const { sessionId } = useLocalSearchParams<{ sessionId: string }>();
  const { detail, isPending, refetch, title, newChat } = useChatScreen(sessionId);

  function handleRetry() {
    void refetch();
  }

  function renderBody() {
    // Data first: a failed background refetch must not unmount a live thread.
    if (detail) {
      return <ChatThread key={sessionId} sessionID={sessionId} detail={detail} />;
    }
    if (isPending) {
      return (
        <View style={styles.center}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      );
    }
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>{"Couldn't open this chat."}</Text>
        <Pressable
          accessibilityRole="button"
          onPress={handleRetry}
          style={({ pressed }) => [styles.retry, pressed && styles.pressed]}
        >
          <Text style={styles.retryLabel}>Try again</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title={title}
        right={<IconButton icon={PencilEdit02Icon} label="New chat" onPress={newChat} />}
      />
      {renderBody()}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
    padding: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  retry: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceRaised,
  },
  pressed: { opacity: 0.6 },
  retryLabel: { ...type.label, color: colors.text },
});
