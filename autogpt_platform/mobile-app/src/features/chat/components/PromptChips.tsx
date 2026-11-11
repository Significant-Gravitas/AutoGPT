import { Pressable, ScrollView, StyleSheet, Text } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

interface Props {
  prompts: string[];
  onPick: (prompt: string) => void;
}

export function PromptChips({ prompts, onPick }: Props) {
  const styles = useStyles();
  return (
    <ScrollView
      horizontal
      keyboardShouldPersistTaps="handled"
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}
    >
      {prompts.map((prompt) => (
        <Pressable
          key={prompt}
          accessibilityRole="button"
          onPress={() => onPick(prompt)}
          style={({ pressed }) => [styles.chip, pressed && styles.pressed]}
        >
          <Text numberOfLines={2} style={styles.label}>
            {prompt}
          </Text>
        </Pressable>
      ))}
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  row: { gap: spacing.sm, paddingHorizontal: spacing.md },
  chip: {
    maxWidth: 220,
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.label, color: colors.textSecondary },
}));
