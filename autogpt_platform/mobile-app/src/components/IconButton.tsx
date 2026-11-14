import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable } from "react-native";

import { createStyles, radius, useColors, useScheme } from "@/lib/theme";

import { Icon } from "./Icon";

interface Props {
  icon: IconSvgElement;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  variant?: "tray" | "icon" | "primary";
}

export function IconButton({
  icon,
  label,
  onPress,
  disabled,
  variant = "tray",
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  const isPrimary = variant === "primary";

  function iconColor() {
    if (!isPrimary) return colors.text;
    return disabled ? colors.onActionDisabled : colors.onAction;
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled: !!disabled }}
      disabled={disabled}
      hitSlop={8}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        variant === "icon" && styles.icon,
        variant === "icon" && scheme === "light" && styles.shadow,
        isPrimary && styles.primary,
        isPrimary && disabled && styles.primaryDisabled,
        pressed && styles.pressed,
        disabled && !isPrimary && styles.disabled,
      ]}
    >
      <Icon icon={icon} size={20} color={iconColor()} />
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
  icon: {
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.borderBase,
  },
  shadow: {
    shadowColor: "#000000",
    shadowOpacity: 0.05,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  primary: { backgroundColor: colors.action },
  primaryDisabled: { backgroundColor: colors.actionDisabled },
  pressed: { opacity: 0.6 },
  disabled: { opacity: 0.35 },
}));
