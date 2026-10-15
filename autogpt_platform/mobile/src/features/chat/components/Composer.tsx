import * as Haptics from "expo-haptics";
import { useState } from "react";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";

import { Icon } from "../../../design/components/Icon";
import {
  colors,
  MIN_TOUCH,
  radius,
  spacing,
  type,
} from "../../../design/theme";

interface Props {
  placeholder: string;
  /** Why nothing can be sent at all (e.g. a let-go expert). Hides the input. */
  readOnlyReason: string | null;
  /** A reply is in progress: sending waits for it, and Stop is offered. */
  busy: boolean;
  canStop: boolean;
  bottomInset: number;
  onSend: (text: string) => void;
  onStop: () => void;
}

const MAX_INPUT_HEIGHT = 160;

export function Composer({
  placeholder,
  readOnlyReason,
  busy,
  canStop,
  bottomInset,
  onSend,
  onStop,
}: Props) {
  const [text, setText] = useState("");
  const canSend = !busy && text.trim().length > 0;

  function send() {
    if (!canSend) return;
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    onSend(text);
    setText("");
  }

  if (readOnlyReason) {
    return (
      <View style={[styles.bar, { paddingBottom: bottomInset + spacing.sm }]}>
        <Text style={[type.footnote, styles.readOnly]}>{readOnlyReason}</Text>
      </View>
    );
  }

  return (
    <View style={[styles.bar, { paddingBottom: bottomInset + spacing.sm }]}>
      <View style={styles.field}>
        <TextInput
          value={text}
          onChangeText={setText}
          placeholder={placeholder}
          placeholderTextColor={colors.placeholder}
          multiline
          style={[type.body, styles.input]}
          accessibilityLabel="Message"
          accessibilityHint={
            busy ? "You can send when the current reply finishes" : undefined
          }
          testID="composer-input"
        />
        {busy && canStop ? (
          <Pressable
            onPress={onStop}
            accessibilityRole="button"
            accessibilityLabel="Stop reply"
            hitSlop={6}
            style={styles.button}
            testID="composer-stop"
          >
            <Icon name="stop.circle.fill" size={32} color={colors.label} />
          </Pressable>
        ) : (
          <Pressable
            onPress={send}
            disabled={!canSend}
            accessibilityRole="button"
            accessibilityLabel="Send"
            accessibilityState={{ disabled: !canSend }}
            hitSlop={6}
            style={styles.button}
            testID="composer-send"
          >
            <Icon
              name="arrow.up.circle.fill"
              size={32}
              color={canSend ? colors.tint : colors.tertiaryLabel}
            />
          </Pressable>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.separator,
  },
  field: {
    flexDirection: "row",
    alignItems: "flex-end",
    borderRadius: radius.bubble + 2,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.separator,
    backgroundColor: colors.assistantBubble,
    paddingLeft: spacing.md,
    paddingRight: spacing.xs,
  },
  input: {
    flex: 1,
    minHeight: MIN_TOUCH,
    maxHeight: MAX_INPUT_HEIGHT,
    color: colors.label,
    paddingTop: spacing.md - 1,
    paddingBottom: spacing.md - 1,
  },
  button: {
    width: MIN_TOUCH,
    height: MIN_TOUCH,
    alignItems: "center",
    justifyContent: "center",
  },
  readOnly: {
    color: colors.secondaryLabel,
    textAlign: "center",
    paddingVertical: spacing.md,
  },
});
