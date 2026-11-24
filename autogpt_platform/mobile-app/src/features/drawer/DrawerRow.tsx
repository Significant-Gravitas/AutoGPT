import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, Text } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  label: string;
  onPress: () => void;
  icon?: IconSvgElement;
}

export function DrawerRow({ label, onPress, icon }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      {icon ? <Icon icon={icon} size={20} color={colors.textSecondary} /> : null}
      <Text numberOfLines={1} style={styles.label}>
        {label}
      </Text>
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
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.label, flex: 1, color: colors.text },
}));
