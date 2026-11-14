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
 * chat, all of that expert's documents. With nothing to list at all, one
 * empty state stands in for both.
 */
export function FileLibrary({ sessionID, expert, isOpen }: Props) {
  const panel = useFilesPanel({ sessionID, expertID: expert?.id ?? null, isOpen });
  const styles = useStyles();
  const { session } = panel;
  // In an expert's chat only the documents section shows the wait.
  const showSession =
    session.isError || session.files.length > 0 || (session.isLoading && !expert);

  const documents = panel.expert;
  const hasNothingFromExpert =
    !!expert &&
    !!documents &&
    !showSession &&
    !documents.isLoading &&
    !documents.isError &&
    documents.files.length === 0;

  return (
    <View style={styles.library}>
      {!showSession && !expert ? <FilesEmptyState /> : null}
      {hasNothingFromExpert ? <FilesEmptyState expertName={expert?.name} /> : null}
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
      {expert && panel.expert && !hasNothingFromExpert ? (
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
