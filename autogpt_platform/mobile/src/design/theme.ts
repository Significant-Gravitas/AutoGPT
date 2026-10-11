import { Color } from "expo-router";
import {
  DynamicColorIOS,
  Platform,
  StyleSheet,
  type ColorValue,
  type TextStyle,
} from "react-native";

const isIOS = Platform.OS === "ios";

function pick(ios: () => ColorValue, android: () => ColorValue): ColorValue {
  return isIOS ? ios() : android();
}

/** The AutoGPT purple (design tokens purple-600 / purple-300) for icons, links and controls. */
const brand = isIOS
  ? DynamicColorIOS({ light: "#6c2edf", dark: "#a476f8" })
  : Color.android.dynamic.primary;

/**
 * Filled brand surfaces under white text (purple-600 / purple-500): at least
 * 5.8:1 contrast in both appearances, where the lighter dark-mode tint is not.
 */
const prominent = isIOS
  ? DynamicColorIOS({ light: "#6c2edf", dark: "#7733f5" })
  : Color.android.dynamic.primary;

/**
 * Semantic system colors, so light/dark, increased contrast and the system
 * accent behave as they do in first-party apps. Brand color is reserved for
 * controls, the user's bubbles and selection.
 */
export const colors = {
  tint: brand,
  prominent,
  onProminent: (isIOS
    ? "#ffffff"
    : Color.android.dynamic.onPrimary) as ColorValue,
  background: pick(
    () => Color.ios.systemBackground,
    () => Color.android.dynamic.surface,
  ),
  groupedBackground: pick(
    () => Color.ios.systemGroupedBackground,
    () => Color.android.dynamic.surfaceContainerLowest,
  ),
  cell: pick(
    () => Color.ios.secondarySystemGroupedBackground,
    () => Color.android.dynamic.surfaceContainerLow,
  ),
  label: pick(
    () => Color.ios.label,
    () => Color.android.dynamic.onSurface,
  ),
  secondaryLabel: pick(
    () => Color.ios.secondaryLabel,
    () => Color.android.dynamic.onSurfaceVariant,
  ),
  tertiaryLabel: pick(
    () => Color.ios.tertiaryLabel,
    () => Color.android.dynamic.onSurfaceVariant,
  ),
  placeholder: pick(
    () => Color.ios.placeholderText,
    () => Color.android.dynamic.onSurfaceVariant,
  ),
  separator: pick(
    () => Color.ios.separator,
    () => Color.android.dynamic.outlineVariant,
  ),
  fill: pick(
    () => Color.ios.tertiarySystemFill,
    () => Color.android.dynamic.surfaceContainerHigh,
  ),
  assistantBubble: pick(
    () => Color.ios.secondarySystemBackground,
    () => Color.android.dynamic.surfaceContainer,
  ),
  danger: pick(
    () => Color.ios.systemRed,
    () => Color.android.dynamic.error,
  ),
  success: pick(
    () => Color.ios.systemGreen,
    () => Color.android.dynamic.primary,
  ),
  warning: pick(
    () => Color.ios.systemOrange,
    () => Color.android.dynamic.error,
  ),
  link: pick(
    () => Color.ios.link,
    () => Color.android.dynamic.primary,
  ),
};

/** iOS Human Interface Guidelines text styles at the default content size. */
export const type = StyleSheet.create({
  largeTitle: { fontSize: 34, lineHeight: 41, fontWeight: "700" },
  title2: { fontSize: 22, lineHeight: 28, fontWeight: "700" },
  title3: { fontSize: 20, lineHeight: 25, fontWeight: "600" },
  headline: { fontSize: 17, lineHeight: 22, fontWeight: "600" },
  body: { fontSize: 17, lineHeight: 22 },
  callout: { fontSize: 16, lineHeight: 21 },
  subheadline: { fontSize: 15, lineHeight: 20 },
  footnote: { fontSize: 13, lineHeight: 18 },
  caption: { fontSize: 12, lineHeight: 16 },
  mono: {
    fontSize: 14,
    lineHeight: 19,
    fontFamily: Platform.select({ ios: "Menlo", default: "monospace" }),
  },
}) satisfies Record<string, TextStyle>;

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };

export const radius = { sm: 8, md: 12, bubble: 20, pill: 999 };

/** Minimum touch target from the HIG. */
export const MIN_TOUCH = 44;
