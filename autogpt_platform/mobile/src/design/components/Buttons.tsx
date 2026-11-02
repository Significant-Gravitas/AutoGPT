import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  type StyleProp,
  type ViewStyle,
} from "react-native";

import { colors, MIN_TOUCH, radius, spacing, type } from "../theme";

interface Props {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  busy?: boolean;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}

/** The single prominent action on a screen. */
export function PrimaryButton({
  title,
  onPress,
  disabled,
  busy,
  style,
  testID,
}: Props) {
  const inactive = disabled || busy;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      accessibilityState={{ disabled: !!inactive, busy: !!busy }}
      disabled={inactive}
      onPress={onPress}
      testID={testID}
      style={({ pressed }) => [
        styles.primary,
        inactive && styles.inactive,
        pressed && styles.pressed,
        style,
      ]}
    >
      {busy ? (
        <ActivityIndicator color={colors.onProminent} />
      ) : (
        <Text style={[type.headline, styles.primaryText]}>{title}</Text>
      )}
    </Pressable>
  );
}

/** A secondary text action, styled like a system plain button. */
export function PlainButton({
  title,
  onPress,
  disabled,
  style,
  testID,
}: Omit<Props, "busy">) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      accessibilityState={{ disabled: !!disabled }}
      disabled={disabled}
      onPress={onPress}
      testID={testID}
      hitSlop={8}
      style={({ pressed }) => [
        styles.plain,
        pressed && styles.pressed,
        disabled && styles.inactive,
        style,
      ]}
    >
      <Text style={[type.body, styles.plainText]}>{title}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  primary: {
    minHeight: 50,
    borderRadius: radius.pill,
    backgroundColor: colors.prominent,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.xl,
  },
  primaryText: { color: colors.onProminent },
  plain: {
    minHeight: MIN_TOUCH,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.md,
  },
  plainText: { color: colors.tint },
  inactive: { opacity: 0.45 },
  pressed: { opacity: 0.7 },
});
