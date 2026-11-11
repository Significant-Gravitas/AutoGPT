import { useRef, useState } from "react";
import { View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing } from "@/lib/theme";

import { isUploading, type Attachment } from "../attachmentHelpers";
import { AttachmentChips } from "./AttachmentChips";
import { ComposerField } from "./ComposerField";

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

  return (
    <View style={[styles.wrap, { paddingBottom: insets.bottom + spacing.sm }]}>
      {attachments.length > 0 ? (
        <AttachmentChips
          attachments={attachments}
          onRemove={onRemoveAttachment}
          onRetry={onRetryAttachment}
        />
      ) : null}
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
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  wrap: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
  },
}));
