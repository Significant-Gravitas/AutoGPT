import { ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { StyleSheet, TextInput, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import { colors, radius, spacing, type } from "@/lib/theme";

interface Props {
  onSend: (text: string) => void;
  onStop?: () => void;
  isBusy?: boolean;
  isDisabled?: boolean;
  placeholder?: string;
}

export function Composer({
  onSend,
  onStop,
  isBusy,
  isDisabled,
  placeholder = "Ask Otto anything",
}: Props) {
  const [text, setText] = useState("");
  const insets = useSafeAreaInsets();
  const canSend = text.trim().length > 0 && !isBusy && !isDisabled;

  function handleSend() {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  }

  return (
    <View style={[styles.wrap, { paddingBottom: insets.bottom + spacing.sm }]}>
      <View style={styles.field}>
        <TextInput
          accessibilityLabel="Message"
          multiline
          value={text}
          onChangeText={setText}
          editable={!isDisabled}
          placeholder={placeholder}
          placeholderTextColor={colors.textTertiary}
          selectionColor={colors.brand}
          style={styles.input}
        />
        {isBusy && onStop ? (
          <IconButton icon={StopIcon} label="Stop" onPress={onStop} filled />
        ) : (
          <IconButton
            icon={ArrowUp02Icon}
            label="Send"
            onPress={handleSend}
            disabled={!canSend}
            filled
          />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
  },
  field: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: spacing.sm,
    paddingLeft: spacing.lg,
    paddingRight: spacing.xs,
    paddingVertical: spacing.xs,
    borderRadius: radius.lg + 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  input: {
    ...type.body,
    flex: 1,
    maxHeight: 140,
    paddingTop: 9,
    paddingBottom: 9,
    color: colors.text,
  },
});
