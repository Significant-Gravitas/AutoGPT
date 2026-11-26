import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { BriefingFrequency } from "@/lib/api/preferences";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { BRIEFING_OPTIONS } from "../notificationHelpers";

interface Props {
  value: BriefingFrequency;
  onChoose: (frequency: BriefingFrequency) => void;
}

/** How often the Briefing email comes. */
export function BriefingPicker({ value, onChoose }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <View>
      <View style={styles.heading}>
        <Text style={styles.label}>Briefing</Text>
        <Text style={styles.description}>
          What your workflows got done, at around 07:30 your time. Never sent when nothing ran.
        </Text>
      </View>
      <View accessibilityRole="radiogroup">
        {BRIEFING_OPTIONS.map((option) => {
          const isSelected = option.value === value;
          return (
            <Pressable
              key={option.value}
              accessibilityRole="radio"
              accessibilityLabel={option.label}
              accessibilityState={{ checked: isSelected }}
              onPress={() => onChoose(option.value)}
              style={({ pressed }) => [styles.row, pressed && styles.pressed]}
            >
              <Text style={styles.option}>{option.label}</Text>
              {isSelected ? <Icon icon={Tick02Icon} size={20} color={colors.text} /> : null}
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  heading: {
    gap: 2,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  label: { ...type.body, color: colors.text },
  description: { ...type.small, color: colors.textSecondary },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 48,
    paddingHorizontal: spacing.lg,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.panelDivider,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  option: { ...type.body, flex: 1, color: colors.text },
}));
