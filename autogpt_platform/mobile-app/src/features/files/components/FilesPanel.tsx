import { ScrollView, StyleSheet, Text } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { colors, spacing, type } from "@/lib/theme";

import { useFilesPanel } from "../useFilesPanel";
import { FileSection } from "./FileSection";
import { FileViewer } from "./FileViewer";

interface Props {
  sessionID: string;
  expertID: string | null;
  expertName: string | null;
  isOpen: boolean;
}

export function FilesPanel({ sessionID, expertID, expertName, isOpen }: Props) {
  const { session, expert, openFile, selectFile, closeFile } = useFilesPanel({
    sessionID,
    expertID,
    isOpen,
  });
  const insets = useSafeAreaInsets();

  return (
    <>
      <ScrollView
        contentContainerStyle={[
          styles.content,
          { paddingTop: insets.top + spacing.md, paddingBottom: insets.bottom + spacing.xl },
        ]}
      >
        <Text accessibilityRole="header" style={styles.title}>
          Files
        </Text>
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
      </ScrollView>
      <FileViewer file={openFile} onClose={closeFile} />
    </>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.xl, paddingHorizontal: spacing.sm },
  title: { ...type.heading, color: colors.text, paddingHorizontal: spacing.md },
});
