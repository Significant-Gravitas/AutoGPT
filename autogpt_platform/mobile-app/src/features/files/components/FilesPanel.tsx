import { ScrollView, Text } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing, type } from "@/lib/theme";

import { FileLibrary } from "./FileLibrary";

interface Props {
  sessionID: string;
  isOpen: boolean;
}

/** The panel in a chat with Otto: files only. */
export function FilesPanel({ sessionID, isOpen }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();

  return (
    <ScrollView
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + spacing.md, paddingBottom: insets.bottom + spacing.xl },
      ]}
    >
      <Text accessibilityRole="header" style={styles.title}>
        Files
      </Text>
      <FileLibrary
        sessionID={sessionID}
        expertID={null}
        expertName={null}
        isOpen={isOpen}
      />
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  content: { gap: spacing.xl, paddingHorizontal: spacing.sm },
  title: { ...type.heading, color: colors.text, paddingHorizontal: spacing.md },
}));
