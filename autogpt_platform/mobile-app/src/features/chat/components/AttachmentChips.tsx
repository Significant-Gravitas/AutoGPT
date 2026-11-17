import { ScrollView } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

import type { Attachment } from "../attachmentHelpers";
import { AttachmentChip } from "./AttachmentChip";

interface Props {
  attachments: Attachment[];
  onRemove?: (id: string) => void;
  onRetry?: (id: string) => void;
}

export function AttachmentChips({ attachments, onRemove, onRetry }: Props) {
  const styles = useStyles();
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={styles.row}
      style={styles.scroll}
    >
      {attachments.map((attachment) => (
        <AttachmentChip
          key={attachment.id}
          attachment={attachment}
          onRemove={onRemove}
          onRetry={onRetry}
        />
      ))}
    </ScrollView>
  );
}

const useStyles = createStyles(() => ({
  scroll: { flexGrow: 0, marginBottom: spacing.sm },
  row: { gap: spacing.sm },
}));
