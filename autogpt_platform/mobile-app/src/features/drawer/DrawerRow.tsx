import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  label: string;
  onPress: () => void;
  icon?: IconSvgElement;
  isActive?: boolean;
  isBusy?: boolean;
}

export function DrawerRow({ label, onPress, icon, isActive, isBusy }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: isActive }}
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        isActive && styles.active,
        pressed && styles.pressed,
      ]}
    >
      {icon ? <Icon icon={icon} size={20} color={colors.textSecondary} /> : null}
      <Text numberOfLines={1} style={styles.label}>
        {label}
      </Text>
      {isBusy ? <View style={styles.dot} /> : null}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 44,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  active: { backgroundColor: colors.surfaceRaised },
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.label, flex: 1, color: colors.text },
  dot: {
    width: 8,
    height: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.brand,
  },
}));
