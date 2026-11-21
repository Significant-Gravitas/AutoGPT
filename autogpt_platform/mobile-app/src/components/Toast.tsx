import { AlertCircleIcon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text } from "react-native";
import Animated, { FadeInDown, FadeOutDown } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { Icon } from "./Icon";
import type { ToastState } from "./useToast";

interface Props {
  toast: ToastState;
}

export function Toast({ toast }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();
  const isError = toast.tone === "error";

  return (
    <Animated.View
      entering={FadeInDown.duration(200)}
      exiting={FadeOutDown.duration(160)}
      accessibilityLiveRegion="polite"
      style={[styles.toast, { bottom: insets.bottom + spacing.md }]}
    >
      <Icon
        icon={isError ? AlertCircleIcon : CheckmarkCircle02Icon}
        size={18}
        color={isError ? colors.error : colors.success}
      />
      <Text style={styles.message}>{toast.message}</Text>
      {toast.actionLabel && toast.onAction ? (
        <Pressable accessibilityRole="button" hitSlop={8} onPress={toast.onAction}>
          <Text style={styles.action}>{toast.actionLabel}</Text>
        </Pressable>
      ) : null}
    </Animated.View>
  );
}

const useStyles = createStyles((colors) => ({
  toast: {
    position: "absolute",
    left: spacing.lg,
    right: spacing.lg,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surfaceRaised,
    shadowColor: "#000000",
    shadowOpacity: 0.2,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  message: { ...type.small, flex: 1, color: colors.text },
  action: { ...type.label, color: colors.accentText },
}));
