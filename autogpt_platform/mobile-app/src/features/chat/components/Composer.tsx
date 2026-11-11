import { ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import { useRef, useState } from "react";
import { StyleSheet, TextInput, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  onSend: (text: string) => boolean | Promise<boolean>;
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
  const styles = useStyles();
  const colors = useColors();
  const [text, setText] = useState("");
  const [isSending, setIsSending] = useState(false);
  // State lands a render late; the ref stops a second press in the same frame.
  const isSendingRef = useRef(false);
  const insets = useSafeAreaInsets();
  const canSend =
    text.trim().length > 0 && !isBusy && !isDisabled && !isSending;

  // The text stays until the send is accepted, so a failure keeps it.
  async function handleSend() {
    if (!canSend || isSendingRef.current) return;
    isSendingRef.current = true;
    setIsSending(true);
    try {
      if (await onSend(text.trim())) setText("");
    } finally {
      isSendingRef.current = false;
      setIsSending(false);
    }
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
          <IconButton icon={StopIcon} label="Stop" onPress={onStop} variant="primary" />
        ) : (
          <IconButton
            icon={ArrowUp02Icon}
            label="Send"
            onPress={handleSend}
            disabled={!canSend}
            variant="primary"
          />
        )}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
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
}));
