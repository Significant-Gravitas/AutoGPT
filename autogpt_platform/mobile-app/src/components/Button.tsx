import type { IconSvgElement } from "@hugeicons/react-native";
import { ActivityIndicator, Pressable, Text } from "react-native";

import {
  createStyles,
  radius,
  useColors,
  useScheme,
  type Colors,
} from "@/lib/theme";

import { Icon } from "./Icon";

type Variant = "primary" | "secondary" | "outline" | "ghost" | "destructive";
type Size = "large" | "small";

interface Props {
  label: string;
  onPress: () => void;
  variant?: Variant;
  size?: Size;
  leadingIcon?: IconSvgElement;
  isLoading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  accessibilityLabel?: string;
}

const ICON_SIZE: Record<Size, number> = { large: 18, small: 16 };

export function Button({
  label,
  onPress,
  variant = "primary",
  size = "large",
  leadingIcon,
  isLoading,
  disabled,
  fullWidth,
  accessibilityLabel,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  const isDisabled = !!disabled;
  const labelColor = buttonLabelColor(variant, isDisabled, colors);

  function renderLeading() {
    if (isLoading) return <ActivityIndicator size="small" color={labelColor} />;
    if (!leadingIcon) return null;
    return <Icon icon={leadingIcon} size={ICON_SIZE[size]} color={labelColor} />;
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      accessibilityState={{ disabled: isDisabled, busy: !!isLoading }}
      disabled={isDisabled || isLoading}
      onPress={onPress}
      style={({ pressed }) => [
        styles.base,
        styles[size],
        styles[variant],
        variant === "secondary" && scheme === "light" && styles.shadow,
        isDisabled && variant === "primary" && styles.primaryDisabled,
        isDisabled && variant === "secondary" && styles.secondaryDisabled,
        fullWidth && styles.fullWidth,
        pressed && styles.pressed,
      ]}
    >
      {renderLeading()}
      <Text
        numberOfLines={1}
        style={[styles.label, styles[`${size}Label`], { color: labelColor }]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

function buttonLabelColor(variant: Variant, isDisabled: boolean, colors: Colors) {
  if (variant === "primary") {
    return isDisabled ? colors.onActionDisabled : colors.onAction;
  }
  if (variant === "secondary" && isDisabled) return colors.textTertiary;
  if (variant === "destructive") return "#FEFEFE";
  return colors.text;
}

const useStyles = createStyles((colors) => ({
  base: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "flex-start",
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  large: { height: 46, paddingHorizontal: 16, gap: 8 },
  small: { height: 36, paddingHorizontal: 12, gap: 6 },
  primary: { backgroundColor: colors.action, borderColor: colors.action },
  primaryDisabled: {
    backgroundColor: colors.actionDisabled,
    borderColor: colors.actionDisabled,
  },
  secondary: {
    backgroundColor: colors.background,
    borderColor: colors.borderBase,
  },
  secondaryDisabled: { backgroundColor: colors.surface },
  outline: { backgroundColor: "transparent", borderColor: colors.borderStrong },
  ghost: { backgroundColor: "transparent", borderColor: "transparent" },
  destructive: { backgroundColor: colors.error, borderColor: colors.error },
  shadow: {
    shadowColor: "#000000",
    shadowOpacity: 0.05,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  fullWidth: { alignSelf: "stretch" },
  pressed: { opacity: 0.85 },
  label: { fontWeight: "500" },
  largeLabel: { fontSize: 15, lineHeight: 20 },
  smallLabel: { fontSize: 14, lineHeight: 20 },
}));
