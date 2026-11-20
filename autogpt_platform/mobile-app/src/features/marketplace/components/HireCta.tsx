import { ActivityIndicator, type LayoutChangeEvent, Pressable, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import Svg, { Defs, LinearGradient, Rect, Stop } from "react-native-svg";

import { createStyles, radius, spacing, useColors } from "@/lib/theme";

interface Props {
  name: string;
  isOnTeam: boolean;
  isHiring: boolean;
  isCheckingTeam: boolean;
  isOpeningChat: boolean;
  error: string | null;
  onHire: () => void;
  onChat: () => void;
  onLayout: (event: LayoutChangeEvent) => void;
}

/** One pill pinned to the foot of the page, over a fade into the page grey. */
export function HireCta({
  name,
  isOnTeam,
  isHiring,
  isCheckingTeam,
  isOpeningChat,
  error,
  onHire,
  onChat,
  onLayout,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const isBusy = isOnTeam ? isOpeningChat : isHiring || isCheckingTeam;
  const label = isOnTeam ? "Open chat" : `Hire ${name}`;

  return (
    <View
      onLayout={onLayout}
      style={[styles.bar, { paddingBottom: Math.max(insets.bottom, spacing.md) + spacing.sm }]}
    >
      <View style={styles.fade}>
        <Svg width="100%" height="100%" preserveAspectRatio="none">
          <Defs>
            <LinearGradient id="ctaFade" x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0" stopColor={colors.panelBackground} stopOpacity={0} />
              <Stop offset="0.35" stopColor={colors.panelBackground} stopOpacity={1} />
              <Stop offset="1" stopColor={colors.panelBackground} stopOpacity={1} />
            </LinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill="url(#ctaFade)" />
        </Svg>
      </View>
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={isOnTeam ? `Open chat with ${name}` : label}
        accessibilityState={{ busy: isBusy, disabled: isBusy }}
        disabled={isBusy}
        onPress={isOnTeam ? onChat : onHire}
        style={({ pressed }) => [styles.pill, pressed && styles.pressed]}
      >
        {isBusy ? (
          <ActivityIndicator size="small" color={colors.onActionStrong} />
        ) : (
          <Text style={styles.label}>{label}</Text>
        )}
      </Pressable>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  bar: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    gap: spacing.sm,
    paddingTop: spacing.lg,
    paddingHorizontal: spacing.lg,
  },
  fade: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, pointerEvents: "none" },
  error: {
    fontSize: 12.5,
    lineHeight: 17,
    color: colors.error,
    backgroundColor: colors.errorSurface,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    overflow: "hidden",
  },
  pill: {
    height: 44,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.actionStrong,
  },
  pressed: { opacity: 0.8 },
  label: { fontSize: 14, lineHeight: 18, fontWeight: "600", color: colors.onActionStrong },
}));
