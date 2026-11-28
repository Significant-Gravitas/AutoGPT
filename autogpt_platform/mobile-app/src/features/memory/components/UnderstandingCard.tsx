import { Pressable, Text, View } from "react-native";

import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { createStyles, spacing, type } from "@/lib/theme";

import type { UnderstandingRow } from "../helpers";

interface Props {
  rows: UnderstandingRow[];
  onForget: () => void;
}

export function UnderstandingCard({ rows, onForget }: Props) {
  const styles = useStyles();

  if (!rows.length) {
    return (
      <SettingsGroup title="What Otto knows about your work">
        <Text style={styles.empty}>Nothing yet. Otto learns as you chat.</Text>
      </SettingsGroup>
    );
  }

  return (
    <SettingsGroup
      title="What Otto knows about your work"
      note="From your onboarding and your chats. Every new chat with Otto starts with this."
    >
      {rows.map((row) => (
        <View key={row.label} accessible style={styles.row}>
          <Text style={styles.label}>{row.label}</Text>
          <Text style={styles.value}>{row.value}</Text>
        </View>
      ))}
      <Pressable
        accessibilityRole="button"
        onPress={onForget}
        style={({ pressed }) => [styles.action, pressed && styles.pressed]}
      >
        <Text style={styles.actionText}>Forget this</Text>
      </Pressable>
    </SettingsGroup>
  );
}

const useStyles = createStyles((colors) => ({
  row: { gap: 2, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  label: { ...type.small, color: colors.textTertiary },
  value: { ...type.body, color: colors.text },
  empty: {
    ...type.body,
    color: colors.textSecondary,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  action: {
    justifyContent: "center",
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  actionText: { ...type.body, color: colors.error },
}));
