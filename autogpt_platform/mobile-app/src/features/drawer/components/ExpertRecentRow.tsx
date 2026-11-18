import { Pressable, Text, View } from "react-native";

import { prefetchSession } from "@/lib/api/chat";
import { queryClient } from "@/lib/session";
import { ExpertAvatar } from "@/components/ExpertAvatar";
import { chatTitle, type ChatGroup } from "@/features/drawer/helpers";
import { createStyles, radius, spacing, type } from "@/lib/theme";
import { shortTime } from "@/lib/time";

import { ROW_INSET } from "../DrawerRow";

interface Props {
  group: ChatGroup;
  isActive: boolean;
  onOpenChat: (sessionID: string) => void;
}

export function ExpertRecentRow({ group, isActive, onOpenChat }: Props) {
  const styles = useStyles();
  const { latest } = group;

  function handlePressIn() {
    prefetchSession(queryClient, latest.id);
  }

  function handlePress() {
    onOpenChat(latest.id);
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: isActive }}
      onPressIn={handlePressIn}
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
        size={28}
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
    minHeight: 56,
    paddingHorizontal: ROW_INSET - 3,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },
  active: { backgroundColor: colors.menuRowActive },
  pressed: { backgroundColor: colors.surfacePressed },
  body: { flex: 1, minWidth: 0 },
  line: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.body, fontSize: 17, lineHeight: 22, flex: 1, color: colors.text },
  time: { ...type.small, color: colors.textSecondary },
  title: { ...type.small, fontSize: 14, color: colors.textSecondary },
  dot: {
    width: 8,
    height: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.brand,
  },
}));
