import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Modal, Platform, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, type } from "@/lib/theme";

import { FileContent } from "./FileContent";

interface Props {
  file: WorkspaceFile | null;
  onClose: () => void;
}

export function FileViewer({ file, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  // A page sheet starts below the status bar on iOS; full screen elsewhere.
  const topInset = Platform.OS === "ios" ? 0 : insets.top;

  return (
    <Modal
      visible={file !== null}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={[styles.screen, { paddingTop: topInset }]}>
        <View style={styles.header}>
          <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
            {file?.name}
          </Text>
          <IconButton icon={Cancel01Icon} label="Close" onPress={onClose} />
        </View>
        {file ? <FileContent key={file.id} file={file} /> : null}
      </View>
    </Modal>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingLeft: spacing.lg,
    paddingRight: spacing.sm,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  title: { ...type.heading, flex: 1, color: colors.text },
}));
