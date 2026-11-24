import { Pressable, ScrollView, Text } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import { type Area, AREAS } from "../helpers";

interface Props {
  selected: Area | null;
  onSelect: (area: Area | null) => void;
}

const OPTIONS: { key: Area | null; label: string }[] = [
  { key: null, label: "All" },
  ...AREAS,
];

export function AreaChips({ selected, onSelect }: Props) {
  const styles = useStyles();
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}
      keyboardShouldPersistTaps="handled"
    >
      {OPTIONS.map((option) => {
        const isOn = option.key === selected;
        return (
          <Pressable
            key={option.label}
            accessibilityRole="button"
            accessibilityState={{ selected: isOn }}
            onPress={() => onSelect(option.key)}
            style={({ pressed }) => [styles.chip, isOn && styles.on, pressed && styles.pressed]}
          >
            <Text style={[styles.label, isOn && styles.labelOn]}>{option.label}</Text>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  row: { gap: spacing.sm, paddingHorizontal: spacing.lg },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  on: { backgroundColor: colors.action, borderColor: colors.action },
  pressed: { opacity: 0.85 },
  label: { ...type.small, fontWeight: "500", color: colors.textSecondary },
  labelOn: { color: colors.onAction },
}));
