import { Pressable, Text, View } from "react-native";

import { prefetchSession, type SessionSummary } from "@/lib/api/chat";
import { queryClient } from "@/lib/session";
import { ExpertAvatar } from "@/components/ExpertAvatar";
import { chatTitle, type ChatMatch } from "@/features/drawer/helpers";
import { createStyles, radius, spacing, type } from "@/lib/theme";
import { shortTime } from "@/lib/time";

import { ROW_INSET } from "../DrawerRow";

interface Props {
  match: ChatMatch;
  isActive: boolean;
  onOpenChat: (sessionID: string) => void;
  onLongPress: (session: SessionSummary) => void;
}

export function SearchResultRow({ match, isActive, onOpenChat, onLongPress }: Props) {
  const styles = useStyles();
  const { session, group } = match;
  const time = shortTime(session.updated_at);

  function handlePressIn() {
    prefetchSession(queryClient, session.id);
  }

  function handlePress() {
    onOpenChat(session.id);
  }

  function handleLongPress() {
    onLongPress(session);
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityHint="Hold for options"
      accessibilityState={{ selected: isActive }}
      onPressIn={handlePressIn}
      onPress={handlePress}
      onLongPress={handleLongPress}
      delayLongPress={350}
      style={({ pressed }) => [
        styles.row,
        isActive && styles.active,
        pressed && styles.pressed,
      ]}
    >
      <ExpertAvatar
        name={group.label}
        source={group.avatar}
        tint={group.tint}
        size={28}
      />
      <View style={styles.body}>
        <Text numberOfLines={1} style={styles.title}>
          {chatTitle(session)}
        </Text>
        <Text numberOfLines={1} style={styles.meta}>
          {time ? `${group.label} · ${time}` : group.label}
        </Text>
      </View>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 56,
    paddingHorizontal: ROW_INSET - 3,
    paddingVertical: spacing.xs,
    borderRadius: radius.md,
  },
  active: { backgroundColor: colors.menuRowActive },
  pressed: { backgroundColor: colors.surfacePressed },
  body: { flex: 1, minWidth: 0 },
  title: { ...type.body, fontSize: 17, lineHeight: 22, color: colors.text },
  meta: { ...type.small, fontSize: 14, color: colors.textSecondary },
}));
