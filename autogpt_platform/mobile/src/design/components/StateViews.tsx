import type { SFSymbol } from "expo-symbols";
import type { ReactNode } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";

import { colors, spacing, type } from "../theme";
import { PlainButton } from "./Buttons";
import { Icon } from "./Icon";

export function LoadingState({ label }: { label: string }) {
  return (
    <View style={styles.center} accessible accessibilityLabel={label}>
      <ActivityIndicator />
    </View>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <View style={styles.center}>
      <Icon
        name="exclamationmark.triangle"
        size={32}
        color={colors.secondaryLabel}
      />
      <Text style={[type.body, styles.message]} accessibilityRole="alert">
        {message}
      </Text>
      {onRetry ? <PlainButton title="Try Again" onPress={onRetry} /> : null}
    </View>
  );
}

export function EmptyState({
  symbol,
  title,
  message,
  action,
}: {
  symbol: SFSymbol;
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <View style={styles.center}>
      <Icon name={symbol} size={40} color={colors.tertiaryLabel} />
      <Text style={[type.title3, styles.title]} accessibilityRole="header">
        {title}
      </Text>
      <Text style={[type.body, styles.message]}>{message}</Text>
      {action}
    </View>
  );
}

/** An inline, non-blocking notice such as "Reconnecting…" or a failed refresh. */
export function InlineNotice({
  text,
  tone = "neutral",
}: {
  text: string;
  tone?: "neutral" | "danger";
}) {
  return (
    <View style={styles.notice} accessibilityLiveRegion="polite">
      <Text
        style={[
          type.footnote,
          { color: tone === "danger" ? colors.danger : colors.secondaryLabel },
        ]}
      >
        {text}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.xxl,
    gap: spacing.md,
  },
  title: { color: colors.label, textAlign: "center" },
  message: { color: colors.secondaryLabel, textAlign: "center" },
  notice: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    alignItems: "center",
  },
});
