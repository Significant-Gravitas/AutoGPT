import { View } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

import { useFilesPanel } from "../useFilesPanel";
import { FileSection } from "./FileSection";
import { FilesEmptyState } from "./FilesEmptyState";
import { FileViewer } from "./FileViewer";

interface Props {
  sessionID: string;
  expert: { id: string; name: string } | null;
  isOpen: boolean;
}

/**
 * As on the web: this chat's files once there are any, then, in an expert's
 * chat, all of that expert's documents. Otto's chat with no files gets one
 * empty state.
 */
export function FileLibrary({ sessionID, expert, isOpen }: Props) {
  const panel = useFilesPanel({ sessionID, expertID: expert?.id ?? null, isOpen });
  const styles = useStyles();
  const { session } = panel;
  // In an expert's chat only the documents section shows the wait.
  const showSession =
    session.isError || session.files.length > 0 || (session.isLoading && !expert);

  return (
    <View style={styles.library}>
      {!showSession && !expert ? <FilesEmptyState /> : null}
      {showSession ? (
        <FileSection
          title="Files in this chat"
          showCount
          files={session.files}
          isLoading={session.isLoading}
          isError={session.isError}
          onRetry={session.retry}
          onSelect={panel.selectFile}
        />
      ) : null}
      {expert && panel.expert ? (
        <FileSection
          title={`All ${expert.name}'s documents`}
          emptyText={`${expert.name} hasn't created any documents yet.`}
          files={panel.expert.files}
          isLoading={panel.expert.isLoading}
          isError={panel.expert.isError}
          onRetry={panel.expert.retry}
          onSelect={panel.selectFile}
        />
      ) : null}
      <FileViewer file={panel.openFile} onClose={panel.closeFile} />
    </View>
  );
}

const useStyles = createStyles(() => ({
  library: { gap: spacing.xl },
}));
