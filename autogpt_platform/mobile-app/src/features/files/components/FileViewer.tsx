import { Cancel01Icon, Share08Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Modal, Platform, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { fileDetailLine } from "../listHelpers";
import { useShareFile } from "../useShareFile";
import { FileContent } from "./FileContent";

interface Props {
  file: WorkspaceFile | null;
  onClose: () => void;
}

export function FileViewer({ file, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();
  const { share, sharingID } = useShareFile();
  // A page sheet starts below the status bar on iOS; full screen elsewhere.
  const topInset = Platform.OS === "ios" ? 0 : insets.top;
  const isSharing = !!file && sharingID === file.id;

  function handleShare() {
    if (file) void share(file);
  }

  return (
    <Modal
      visible={file !== null}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={[styles.screen, { paddingTop: topInset }]}>
        <View style={styles.header}>
          <View style={styles.titles}>
            <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
              {file?.name}
            </Text>
            {file ? <Text style={styles.detail}>{fileDetailLine(file)}</Text> : null}
          </View>
          {isSharing ? (
            <View style={styles.busy}>
              <ActivityIndicator size="small" color={colors.textTertiary} />
            </View>
          ) : (
            <IconButton icon={Share08Icon} label="Share" onPress={handleShare} disabled={!file} />
          )}
          <IconButton icon={Cancel01Icon} label="Close" onPress={onClose} />
        </View>
        {file ? (
          <FileContent key={file.id} file={file} isSharing={isSharing} onOpenIn={handleShare} />
        ) : null}
      </View>
    </Modal>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingLeft: spacing.lg,
    paddingRight: spacing.sm,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  titles: { flex: 1, minWidth: 0 },
  title: { fontSize: 16, lineHeight: 21, fontWeight: "600", color: colors.text },
  detail: { fontSize: 12.5, lineHeight: 16, color: colors.textTertiary },
  busy: { width: 40, height: 40, alignItems: "center", justifyContent: "center" },
}));
