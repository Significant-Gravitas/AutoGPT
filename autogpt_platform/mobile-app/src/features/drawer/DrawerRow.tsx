import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, Text } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

// Lines a row's icon up under the drawer's title.
export const ROW_INSET = 20;

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
      {icon ? <Icon icon={icon} size={22} color={colors.text} /> : null}
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
    gap: spacing.lg,
    minHeight: 48,
    paddingHorizontal: ROW_INSET,
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.body, fontSize: 17, lineHeight: 22, flex: 1, color: colors.text },
}));
