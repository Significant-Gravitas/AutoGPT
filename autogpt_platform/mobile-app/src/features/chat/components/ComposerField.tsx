import { Add01Icon, ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { StyleSheet, TextInput, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  text: string;
  onChangeText: (text: string) => void;
  placeholder: string;
  canSend: boolean;
  onSend: () => void;
  onStop?: () => void;
  onAttach?: () => void;
  isBusy?: boolean;
  isDisabled?: boolean;
  /** Buttons between the field and the send button. */
  trailing?: ReactNode;
  /** `card` puts the text above a row of buttons, with no outline of its own. */
  layout?: "bar" | "card";
}

export function ComposerField({
  text,
  onChangeText,
  placeholder,
  canSend,
  onSend,
  onStop,
  onAttach,
  isBusy,
  isDisabled,
  trailing,
  layout = "bar",
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const isCard = layout === "card";

  const attach = onAttach ? (
    <IconButton
      icon={Add01Icon}
      label="Attach"
      onPress={onAttach}
      disabled={isDisabled}
      variant="icon"
    />
  ) : null;
  const input = (
    <TextInput
      accessibilityLabel="Message"
      multiline
      value={text}
      onChangeText={onChangeText}
      editable={!isDisabled}
      placeholder={placeholder}
      placeholderTextColor={colors.textTertiary}
      selectionColor={colors.brand}
      style={[styles.input, isCard && styles.cardInput]}
    />
  );
  const send =
    isBusy && onStop ? (
      <IconButton icon={StopIcon} label="Stop" onPress={onStop} variant="primary" />
    ) : (
      <IconButton
        icon={ArrowUp02Icon}
        label="Send"
        onPress={onSend}
        disabled={!canSend}
        variant="primary"
      />
    );

  if (isCard) {
    return (
      <View>
        {input}
        <View style={styles.cardRow}>
          {attach}
          <View style={styles.spacer} />
          {trailing}
          {send}
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.field, onAttach && styles.withAttach]}>
      {attach}
      {input}
      {trailing}
      {send}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
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
  withAttach: { paddingLeft: spacing.xs },
  input: {
    ...type.body,
    flex: 1,
    maxHeight: 140,
    paddingTop: 9,
    paddingBottom: 9,
    color: colors.text,
  },
  // About three lines tall, growing to about six.
  cardInput: {
    flex: 0,
    minHeight: 84,
    maxHeight: 156,
    paddingTop: 14,
    paddingBottom: spacing.xs,
    paddingHorizontal: spacing.lg,
    textAlignVertical: "top",
  },
  cardRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.sm,
  },
  spacer: { flex: 1 },
}));
