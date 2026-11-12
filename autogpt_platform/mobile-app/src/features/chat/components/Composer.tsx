import { Mic01Icon } from "@hugeicons/core-free-icons";
import { useRef, useState } from "react";
import { Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { isLiquidGlassAvailable } from "expo-glass-effect";

import { IconButton } from "@/components/IconButton";
import { tapSend } from "@/lib/haptics";
import { createStyles, spacing, type, useScheme } from "@/lib/theme";

import { isUploading, type Attachment } from "../attachmentHelpers";
import { appendDictation } from "../dictationHelpers";
import { useDictation } from "../useDictation";
import { AttachmentChips } from "./AttachmentChips";
import { ComposerField } from "./ComposerField";
import { RecordingBar } from "./RecordingBar";

interface Props {
  onSend: (text: string) => boolean | Promise<boolean>;
  onStop?: () => void;
  isBusy?: boolean;
  isDisabled?: boolean;
  placeholder?: string;
  onAttach?: () => void;
  attachments?: Attachment[];
  onRemoveAttachment?: (id: string) => void;
  onRetryAttachment?: (id: string) => void;
  /** `bar` is one row for a chat; `card` stacks the text over its buttons, for Home. */
  layout?: "bar" | "card";
}

export function Composer({
  onSend,
  onStop,
  isBusy,
  isDisabled,
  placeholder = "Ask Otto anything",
  onAttach,
  attachments = [],
  onRemoveAttachment,
  onRetryAttachment,
  layout = "bar",
}: Props) {
  const styles = useStyles();
  const scheme = useScheme();
  const isCard = layout === "card";
  const [text, setText] = useState("");
  const [isSending, setIsSending] = useState(false);
  // State lands a render late; the ref stops a second press in the same frame.
  const isSendingRef = useRef(false);
  const insets = useSafeAreaInsets();
  const dictation = useDictation({
    getDraft: () => text,
    onText: (dictated) => setText((prev) => appendDictation(prev, dictated)),
  });
  const hasReadyFile = attachments.some((item) => item.status === "ready");
  const canSend =
    (text.trim().length > 0 || hasReadyFile) &&
    !isUploading(attachments) &&
    !isBusy &&
    !isDisabled &&
    !isSending;

  // The text stays until the send is accepted, so a failure keeps it.
  async function handleSend() {
    if (!canSend || isSendingRef.current) return;
    isSendingRef.current = true;
    setIsSending(true);
    try {
      if (await onSend(text.trim())) {
        tapSend();
        setText("");
      }
    } finally {
      isSendingRef.current = false;
      setIsSending(false);
    }
  }

  function handleDictate() {
    void dictation.start();
  }

  function handleCancelDictation() {
    void dictation.cancel();
  }

  function handleUseDictation() {
    void dictation.confirm();
  }

  const body = dictation.isActive ? (
    <RecordingBar
      recorder={dictation.recorder}
      isTranscribing={dictation.isTranscribing}
      onCancel={handleCancelDictation}
      onConfirm={handleUseDictation}
    />
  ) : (
    <ComposerField
      layout={layout}
      text={text}
      onChangeText={setText}
      placeholder={placeholder}
      canSend={canSend}
      onSend={handleSend}
      onStop={onStop}
      onAttach={onAttach}
      isBusy={isBusy}
      isDisabled={isDisabled}
      trailing={
        isBusy ? null : (
          <IconButton
            icon={Mic01Icon}
            label="Dictate"
            onPress={handleDictate}
            disabled={isDisabled}
          />
        )
      }
    />
  );

  return (
    <View
      style={[
        styles.wrap,
        isCard ? styles.cardWrap : { paddingBottom: insets.bottom + spacing.sm },
        !isCard && isLiquidGlassAvailable() && styles.clearWrap,
      ]}
    >
      {dictation.error ? <Text style={styles.error}>{dictation.error}</Text> : null}
      {attachments.length > 0 ? (
        <AttachmentChips
          attachments={attachments}
          onRemove={onRemoveAttachment}
          onRetry={onRetryAttachment}
        />
      ) : null}
      {isCard ? (
        <View style={[styles.card, scheme === "light" && styles.shadow]}>
          {dictation.isActive ? <View style={styles.recording}>{body}</View> : body}
        </View>
      ) : (
        body
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  wrap: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
  },
  cardWrap: { paddingHorizontal: 0, paddingTop: 0, backgroundColor: "transparent" },
  clearWrap: { backgroundColor: "transparent" },
  card: {
    borderRadius: 24,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  shadow: {
    shadowColor: "#000000",
    shadowOpacity: 0.05,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  // The card keeps its height while the recording bar stands in for the text.
  recording: { minHeight: 140, justifyContent: "flex-end", padding: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.sm,
  },
}));
