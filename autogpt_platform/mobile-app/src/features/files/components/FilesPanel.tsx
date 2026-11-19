import { ScrollView, Text } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { PanelTopBar } from "@/features/expertPanel/components/PanelTopBar";
import { createStyles, spacing, type } from "@/lib/theme";

import { FileLibrary } from "./FileLibrary";

interface Props {
  sessionID: string;
  isOpen: boolean;
  onClose: () => void;
}

/** The panel in a chat with Otto: files only. */
export function FilesPanel({ sessionID, isOpen, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();

  return (
    <ScrollView
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + spacing.xs, paddingBottom: insets.bottom + spacing.xl },
      ]}
    >
      <PanelTopBar sessionID={sessionID} onClose={onClose} />
      <Text accessibilityRole="header" style={styles.title}>
        Files
      </Text>
      <FileLibrary sessionID={sessionID} expert={null} isOpen={isOpen} />
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  content: { gap: spacing.xl, paddingHorizontal: spacing.sm },
  title: { ...type.heading, color: colors.text, paddingHorizontal: spacing.md },
}));
