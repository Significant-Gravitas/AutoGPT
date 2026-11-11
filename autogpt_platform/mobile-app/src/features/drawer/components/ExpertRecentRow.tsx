import { Pressable, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { chatTitle, type ChatGroup } from "@/features/drawer/helpers";
import { createStyles, radius, spacing, type } from "@/lib/theme";
import { shortTime } from "@/lib/time";

interface Props {
  group: ChatGroup;
  isActive: boolean;
  onOpenChat: (sessionID: string) => void;
}

export function ExpertRecentRow({ group, isActive, onOpenChat }: Props) {
  const styles = useStyles();
  const { latest } = group;

  function handlePress() {
    onOpenChat(latest.id);
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
        size={44}
      />
      <View style={styles.body}>
        <View style={styles.line}>
          <Text numberOfLines={1} style={styles.name}>
            {group.label}
          </Text>
          <Text style={styles.time}>{shortTime(latest.updated_at)}</Text>
        </View>
        <Text numberOfLines={1} style={styles.title}>
          {chatTitle(latest)}
        </Text>
      </View>
      {group.isBusy ? <View style={styles.dot} /> : null}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 64,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },
  active: { backgroundColor: colors.surfaceRaised },
  pressed: { backgroundColor: colors.surfacePressed },
  body: { flex: 1, minWidth: 0, gap: 2 },
  line: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.label, flex: 1, color: colors.text },
  time: { ...type.small, color: colors.textTertiary },
  title: { ...type.small, color: colors.textSecondary },
  dot: {
    width: 8,
    height: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.brand,
  },
}));
