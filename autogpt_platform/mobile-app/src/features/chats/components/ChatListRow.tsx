import { Pressable, StyleSheet, Text, View } from "react-native";

import { chatTitle } from "@/features/drawer/helpers";
import type { SessionSummary } from "@/lib/api/chat";
import { createStyles, radius, spacing } from "@/lib/theme";
import { shortTime } from "@/lib/time";

interface Props {
  session: SessionSummary;
  isFirst: boolean;
  onOpenChat: (sessionID: string) => void;
}

export function ChatListRow({ session, isFirst, onOpenChat }: Props) {
  const styles = useStyles();

  function handlePress() {
    onOpenChat(session.id);
  }

  return (
    <Pressable
      accessibilityRole="button"
      onPress={handlePress}
      style={({ pressed }) => [
        styles.row,
        !isFirst && styles.divider,
        pressed && styles.pressed,
      ]}
    >
      <Text numberOfLines={1} style={styles.title}>
        {chatTitle(session)}
      </Text>
      {session.is_processing ? <View style={styles.dot} /> : null}
      <Text style={styles.time}>{shortTime(session.updated_at)}</Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    minHeight: 48,
    paddingHorizontal: spacing.lg,
  },
  divider: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  title: { flex: 1, fontSize: 15, lineHeight: 20, color: colors.text },
  dot: {
    width: 8,
    height: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.brand,
  },
  time: { fontSize: 12, lineHeight: 16, color: colors.textTertiary },
}));
