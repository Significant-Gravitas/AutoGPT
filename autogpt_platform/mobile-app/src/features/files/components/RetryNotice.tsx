import { Pressable, StyleSheet, Text, View } from "react-native";

import { colors, radius, spacing, type } from "@/lib/theme";

interface Props {
  message: string;
  onRetry: () => void;
  centered?: boolean;
}

export function RetryNotice({ message, onRetry, centered }: Props) {
  return (
    <View style={[styles.notice, centered && styles.centered]}>
      <Text style={styles.message}>{message}</Text>
      <Pressable
        accessibilityRole="button"
        onPress={onRetry}
        style={({ pressed }) => [styles.retry, pressed && styles.pressed]}
      >
        <Text style={styles.retryLabel}>Try again</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  notice: { alignItems: "flex-start", gap: spacing.sm, paddingHorizontal: spacing.md },
  centered: { alignItems: "center" },
  message: { ...type.small, color: colors.textSecondary },
  retry: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceRaised,
  },
  pressed: { opacity: 0.6 },
  retryLabel: { ...type.label, color: colors.text },
});
