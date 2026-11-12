import { Moon02Icon, SmartPhone01Icon, Sun03Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import { useState } from "react";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import {
  readAppearance,
  saveAppearance,
  type AppearanceChoice,
} from "@/lib/appearance";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

const OPTIONS: { value: AppearanceChoice; label: string; icon: IconSvgElement }[] = [
  { value: "system", label: "System", icon: SmartPhone01Icon },
  { value: "light", label: "Light", icon: Sun03Icon },
  { value: "dark", label: "Dark", icon: Moon02Icon },
];

/** Light, dark, or follow the phone; the choice is kept across launches. */
export function AppearancePicker() {
  const styles = useStyles();
  const colors = useColors();
  const [choice, setChoice] = useState(readAppearance);

  function pick(value: AppearanceChoice) {
    setChoice(value);
    saveAppearance(value);
  }

  return (
    <View style={styles.section}>
      <Text style={styles.heading}>Appearance</Text>
      <View accessibilityRole="radiogroup" style={styles.track}>
        {OPTIONS.map((option) => {
          const isSelected = option.value === choice;
          return (
            <Pressable
              key={option.value}
              accessibilityRole="radio"
              accessibilityLabel={option.label}
              accessibilityState={{ selected: isSelected }}
              onPress={() => pick(option.value)}
              style={[styles.segment, isSelected && styles.selected]}
            >
              <Icon
                icon={option.icon}
                size={16}
                color={isSelected ? colors.text : colors.textTertiary}
              />
              <Text style={[styles.label, isSelected && styles.labelSelected]}>
                {option.label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.xs, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  heading: { ...type.small, color: colors.textTertiary },
  track: {
    flexDirection: "row",
    padding: 3,
    gap: 2,
    borderRadius: radius.md,
    backgroundColor: colors.surface,
  },
  segment: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    minHeight: 36,
    borderRadius: radius.sm,
  },
  selected: {
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  label: { ...type.small, fontWeight: "500", color: colors.textTertiary },
  labelSelected: { color: colors.text },
}));
