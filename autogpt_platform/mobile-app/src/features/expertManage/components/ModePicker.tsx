import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { AutopilotMode } from "@/lib/api/expertManage";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { MODE_OPTIONS } from "../helpers";

interface Props {
  /** Null follows the platform default. */
  value: AutopilotMode | null;
  onChoose: (mode: AutopilotMode | null) => void;
}

const DEFAULT_OPTION = {
  value: null,
  label: "Default",
  description: "Starts new chats on the platform's default.",
};

/** How new chats with this expert start: how often it asks before acting. */
export function ModePicker({ value, onChoose }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const options = [...MODE_OPTIONS, DEFAULT_OPTION];

  return (
    <View accessibilityRole="radiogroup">
      {options.map((option, index) => {
        const isSelected = option.value === value;
        return (
          <Pressable
            key={option.label}
            accessibilityRole="radio"
            accessibilityLabel={option.label}
            accessibilityHint={option.description}
            accessibilityState={{ checked: isSelected }}
            onPress={() => onChoose(option.value)}
            style={({ pressed }) => [
              styles.row,
              index > 0 && styles.divided,
              pressed && styles.pressed,
            ]}
          >
            <View style={styles.text}>
              <Text style={styles.label}>{option.label}</Text>
              <Text style={styles.description}>{option.description}</Text>
            </View>
            {isSelected ? <Icon icon={Tick02Icon} size={20} color={colors.text} /> : null}
          </Pressable>
        );
      })}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  divided: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  pressed: { backgroundColor: colors.surfacePressed },
  text: { flex: 1, gap: 2 },
  label: { ...type.body, color: colors.text },
  description: { ...type.small, color: colors.textSecondary },
}));
