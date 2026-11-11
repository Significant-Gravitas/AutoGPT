import { Mic01Icon } from "@hugeicons/core-free-icons";
import { useRef, useState } from "react";
import { Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import { createStyles, spacing, type } from "@/lib/theme";

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
}: Props) {
  const styles = useStyles();
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
      if (await onSend(text.trim())) setText("");
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

  return (
    <View style={[styles.wrap, { paddingBottom: insets.bottom + spacing.sm }]}>
      {dictation.error ? <Text style={styles.error}>{dictation.error}</Text> : null}
      {attachments.length > 0 ? (
        <AttachmentChips
          attachments={attachments}
          onRemove={onRemoveAttachment}
          onRetry={onRetryAttachment}
        />
      ) : null}
      {dictation.isActive ? (
        <RecordingBar
          recorder={dictation.recorder}
          isTranscribing={dictation.isTranscribing}
          onCancel={handleCancelDictation}
          onConfirm={handleUseDictation}
        />
      ) : (
        <ComposerField
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
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.sm,
  },
}));
