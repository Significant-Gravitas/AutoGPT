import { Pressable, StyleSheet, Text, View } from "react-native";

import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing } from "@/lib/theme";

import { fileDetailLine } from "../listHelpers";
import { FileThumbnail } from "./FileThumbnail";

interface Props {
  file: WorkspaceFile;
  isFirst: boolean;
  onPress: (file: WorkspaceFile) => void;
}

export function FileRow({ file, isFirst, onPress }: Props) {
  // "Today" and "Yesterday" move with the clock, so the row is never cached.
  "use no memo";
  const styles = useStyles();

  function handlePress() {
    onPress(file);
  }

  return (
    <Pressable
      accessibilityRole="button"
      onPress={handlePress}
      style={({ pressed }) => [
        styles.row,
        !isFirst && styles.divider,
        pressed && styles.pressed,
      ]}
    >
      <FileThumbnail file={file} />
      <View style={styles.text}>
        <Text numberOfLines={1} style={styles.name}>
          {file.name}
        </Text>
        <Text numberOfLines={1} style={styles.detail}>
          {fileDetailLine(file)}
        </Text>
      </View>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.border },
  pressed: { backgroundColor: colors.surfaceRaised },
  text: { flex: 1, minWidth: 0 },
  name: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.text },
  detail: { fontSize: 13, lineHeight: 18, color: colors.textTertiary },
}));
