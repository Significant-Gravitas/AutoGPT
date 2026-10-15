import { Children, type ReactNode, type Ref } from "react";
import {
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  type TextInputProps,
} from "react-native";

import { colors, MIN_TOUCH, radius, spacing, type } from "../theme";
import { Icon } from "./Icon";

/** An inset grouped section, as in Settings. */
export function Section({
  header,
  footer,
  children,
}: {
  header?: string;
  footer?: string;
  children: ReactNode;
}) {
  const rows = Children.toArray(children);
  return (
    <View style={styles.section}>
      {header ? (
        <Text style={[type.footnote, styles.header]} accessibilityRole="header">
          {header.toUpperCase()}
        </Text>
      ) : null}
      <View style={styles.card}>
        {rows.map((row, index) => (
          <View key={index}>
            {index > 0 ? <View style={styles.separator} /> : null}
            {row}
          </View>
        ))}
      </View>
      {footer ? (
        <Text style={[type.footnote, styles.footer]}>{footer}</Text>
      ) : null}
    </View>
  );
}

interface RowProps {
  title: string;
  subtitle?: string | null;
  leading?: ReactNode;
  trailing?: ReactNode;
  destructive?: boolean;
  showsChevron?: boolean;
  onPress?: () => void;
  onLongPress?: () => void;
  /** Names the long-press action for VoiceOver's actions rotor. */
  longPressLabel?: string;
  accessibilityHint?: string;
  testID?: string;
}

export function Row({
  title,
  subtitle,
  leading,
  trailing,
  destructive,
  showsChevron,
  onPress,
  onLongPress,
  longPressLabel,
  accessibilityHint,
  testID,
}: RowProps) {
  const content = (
    <>
      {leading}
      <View style={styles.rowText}>
        <Text
          style={[
            type.body,
            { color: destructive ? colors.danger : colors.label },
          ]}
        >
          {title}
        </Text>
        {subtitle ? (
          <Text style={[type.subheadline, styles.subtitle]} numberOfLines={2}>
            {subtitle}
          </Text>
        ) : null}
      </View>
      {trailing}
      {showsChevron ? (
        <Icon
          name="chevron.right"
          size={14}
          color={colors.tertiaryLabel}
          weight="semibold"
        />
      ) : null}
    </>
  );
  if (!onPress) {
    return (
      <View
        style={styles.row}
        testID={testID}
        accessible
        accessibilityLabel={[title, subtitle].filter(Boolean).join(", ")}
      >
        {content}
      </View>
    );
  }
  return (
    <Pressable
      onPress={onPress}
      onLongPress={onLongPress}
      accessibilityRole="button"
      accessibilityLabel={[title, subtitle].filter(Boolean).join(", ")}
      accessibilityHint={accessibilityHint}
      accessibilityActions={
        onLongPress
          ? [{ name: "secondary", label: longPressLabel ?? "More" }]
          : undefined
      }
      onAccessibilityAction={(event) => {
        if (event.nativeEvent.actionName === "secondary") onLongPress?.();
      }}
      testID={testID}
      style={({ pressed }) => [
        styles.row,
        pressed && { backgroundColor: colors.fill },
      ]}
    >
      {content}
    </Pressable>
  );
}

export function FieldRow({
  label,
  ref,
  ...input
}: TextInputProps & { label: string; ref?: Ref<TextInput> }) {
  return (
    <View style={styles.field}>
      <Text style={[type.footnote, styles.fieldLabel]}>{label}</Text>
      <TextInput
        ref={ref}
        accessibilityLabel={label}
        placeholderTextColor={colors.placeholder}
        style={[type.body, styles.input]}
        {...input}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  section: { marginHorizontal: spacing.lg, marginTop: spacing.xl },
  header: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.lg,
    marginBottom: spacing.sm,
  },
  footer: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.lg,
    marginTop: spacing.sm,
  },
  card: {
    backgroundColor: colors.cell,
    borderRadius: radius.md,
    overflow: "hidden",
  },
  separator: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: colors.separator,
    marginLeft: spacing.lg,
  },
  row: {
    minHeight: MIN_TOUCH,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  rowText: { flex: 1, gap: 2 },
  subtitle: { color: colors.secondaryLabel },
  field: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xs,
  },
  fieldLabel: { color: colors.secondaryLabel },
  input: {
    color: colors.label,
    minHeight: MIN_TOUCH,
    paddingVertical: spacing.sm,
  },
});
