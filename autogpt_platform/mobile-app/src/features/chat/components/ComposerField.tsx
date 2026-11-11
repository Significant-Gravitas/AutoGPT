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
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={[styles.field, onAttach && styles.withAttach]}>
      {onAttach ? (
        <IconButton
          icon={Add01Icon}
          label="Attach"
          onPress={onAttach}
          disabled={isDisabled}
          variant="icon"
        />
      ) : null}
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
      {trailing}
      {isBusy && onStop ? (
        <IconButton icon={StopIcon} label="Stop" onPress={onStop} variant="primary" />
      ) : (
        <IconButton
          icon={ArrowUp02Icon}
          label="Send"
          onPress={onSend}
          disabled={!canSend}
          variant="primary"
        />
      )}
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
}));
