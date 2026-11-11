import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable } from "react-native";

import { createStyles, radius, useColors } from "@/lib/theme";

import { Icon } from "./Icon";

interface Props {
  icon: IconSvgElement;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  filled?: boolean;
}

export function IconButton({ icon, label, onPress, disabled, filled }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      disabled={disabled}
      hitSlop={8}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        filled && styles.filled,
        pressed && styles.pressed,
        disabled && styles.disabled,
      ]}
    >
      <Icon icon={icon} size={20} color={filled ? colors.onBrand : colors.text} />
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  button: {
    width: 40,
    height: 40,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
  },
  filled: { backgroundColor: colors.brand },
  pressed: { opacity: 0.6 },
  disabled: { opacity: 0.35 },
}));
