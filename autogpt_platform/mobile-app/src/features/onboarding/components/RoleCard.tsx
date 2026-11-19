import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, Text } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  label: string;
  icon: IconSvgElement;
  selected: boolean;
  onPress: () => void;
}

export function RoleCard({ label, icon, selected, onPress }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityState={{ selected }}
      accessibilityLabel={label}
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        selected && styles.selected,
        pressed && styles.pressed,
      ]}
    >
      <Icon
        icon={icon}
        size={20}
        color={selected ? colors.accentText : colors.textSecondary}
      />
      <Text numberOfLines={1} style={[styles.label, selected && styles.selectedLabel]}>
        {label}
      </Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    height: 60,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  // Two points wide, one point less padding: the label does not move.
  selected: {
    borderWidth: 2,
    paddingHorizontal: spacing.md - 1,
    borderColor: colors.brand,
    backgroundColor: colors.brandSoft,
  },
  pressed: { opacity: 0.85 },
  label: { ...type.label, flexShrink: 1, color: colors.text },
  selectedLabel: { color: colors.accentText },
}));
