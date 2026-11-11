import { View } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

import { useFilesPanel } from "../useFilesPanel";
import { FileSection } from "./FileSection";
import { FileViewer } from "./FileViewer";

interface Props {
  sessionID: string;
  expertID: string | null;
  expertName: string | null;
  isOpen: boolean;
}

/** "This chat" files and, in an expert's chat, the expert's own. */
export function FileLibrary({ sessionID, expertID, expertName, isOpen }: Props) {
  const { session, expert, openFile, selectFile, closeFile } = useFilesPanel({
    sessionID,
    expertID,
    isOpen,
  });
  const styles = useStyles();

  return (
    <View style={styles.library}>
      <FileSection
        title="This chat"
        emptyText="No files in this chat yet."
        files={session.files}
        isLoading={session.isLoading}
        isError={session.isError}
        onRetry={session.retry}
        onSelect={selectFile}
      />
      {expert ? (
        <FileSection
          title={`${expertName ?? "Expert"}'s files`}
          emptyText="No files yet."
          files={expert.files}
          isLoading={expert.isLoading}
          isError={expert.isError}
          onRetry={expert.retry}
          onSelect={selectFile}
        />
      ) : null}
      <FileViewer file={openFile} onClose={closeFile} />
    </View>
  );
}

const useStyles = createStyles(() => ({
  library: { gap: spacing.xl },
}));
