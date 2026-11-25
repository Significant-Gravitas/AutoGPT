import { Add01Icon, ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { TextInput, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

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
  /** Stands in for the send button, e.g. dictation while the field is empty. */
  action?: ReactNode;
}

// One line tall, growing to about six. DictationField stands in at this size.
export const inputBox = {
  minHeight: 52,
  maxHeight: 156,
  paddingTop: 16,
  paddingBottom: spacing.xs,
  paddingHorizontal: spacing.lg,
} as const;

/** The text above a row of buttons; the card around it is the composer's. */
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
  action,
}: Props) {
  const styles = useStyles();
  const colors = useColors();

  const attach = onAttach ? (
    <IconButton
      icon={Add01Icon}
      label="Attach"
      onPress={onAttach}
      disabled={isDisabled}
      variant="tray"
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
      style={styles.input}
    />
  );
  const send =
    action ??
    (isBusy && onStop ? (
      <IconButton icon={StopIcon} label="Stop" onPress={onStop} variant="primary" />
    ) : (
      <IconButton
        icon={ArrowUp02Icon}
        label="Send"
        onPress={onSend}
        disabled={!canSend}
        variant="primary"
      />
    ));

  return (
    <View>
      {input}
      <View style={styles.row}>
        {attach}
        <View style={styles.spacer} />
        {trailing}
        {send}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  input: {
    ...type.body,
    ...inputBox,
    color: colors.text,
    textAlignVertical: "top",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.sm,
  },
  spacer: { flex: 1 },
}));
