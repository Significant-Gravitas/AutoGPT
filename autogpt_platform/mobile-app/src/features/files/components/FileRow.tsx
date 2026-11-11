import { Pressable, Text } from "react-native";

import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, radius, spacing, type } from "@/lib/theme";

import { fileDetailLine } from "../helpers";

interface Props {
  file: WorkspaceFile;
  onPress: (file: WorkspaceFile) => void;
}

export function FileRow({ file, onPress }: Props) {
  const styles = useStyles();

  function handlePress() {
    onPress(file);
  }

  return (
    <Pressable
      accessibilityRole="button"
      onPress={handlePress}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      <Text numberOfLines={1} style={styles.name}>
        {file.name}
      </Text>
      <Text numberOfLines={1} style={styles.detail}>
        {fileDetailLine(file)}
      </Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    gap: 2,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  name: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
}));
