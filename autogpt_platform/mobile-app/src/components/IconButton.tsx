import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, StyleSheet } from "react-native";

import { createStyles, radius, shadows, useColors, useScheme } from "@/lib/theme";

import { Icon } from "./Icon";

interface Props {
  icon: IconSvgElement;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  /**
   * `menu` is the drawer's raised circle; `floating` the chat header's, which
   * sits over the thread.
   */
  variant?: "tray" | "icon" | "primary" | "menu" | "floating";
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
  const isLight = scheme === "light";

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
        variant === "icon" && isLight && styles.shadow,
        variant === "menu" && styles.menu,
        variant === "menu" && isLight && styles.menuShadow,
        variant === "floating" && styles.floating,
        variant === "floating" && isLight && styles.floatingShadow,
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
  menu: { backgroundColor: colors.menuButton },
  menuShadow: shadows.menuButton,
  floating: {
    width: 38,
    height: 38,
    backgroundColor: colors.floatingButton,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.floatingRing,
  },
  floatingShadow: shadows.floating,
  primary: { backgroundColor: colors.action },
  primaryDisabled: { backgroundColor: colors.actionDisabled },
  pressed: { opacity: 0.6 },
  disabled: { opacity: 0.35 },
}));
