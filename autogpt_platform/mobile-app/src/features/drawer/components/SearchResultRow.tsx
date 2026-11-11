import { Pressable, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { chatTitle, type ChatMatch } from "@/features/drawer/helpers";
import { createStyles, radius, spacing, type } from "@/lib/theme";
import { shortTime } from "@/lib/time";

interface Props {
  match: ChatMatch;
  isActive: boolean;
  onOpenChat: (sessionID: string) => void;
}

export function SearchResultRow({ match, isActive, onOpenChat }: Props) {
  const styles = useStyles();
  const { session, group } = match;
  const time = shortTime(session.updated_at);

  function handlePress() {
    onOpenChat(session.id);
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: isActive }}
      onPress={handlePress}
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
        size={32}
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
    minHeight: 52,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radius.md,
  },
  active: { backgroundColor: colors.surfaceRaised },
  pressed: { backgroundColor: colors.surfacePressed },
  body: { flex: 1, minWidth: 0 },
  title: { ...type.label, color: colors.text },
  meta: { ...type.small, color: colors.textTertiary },
}));
