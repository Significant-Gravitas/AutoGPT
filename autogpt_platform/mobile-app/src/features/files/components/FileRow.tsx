import { File01Icon, Image01Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { fileDetailLine, fileKind } from "../helpers";

interface Props {
  file: WorkspaceFile;
  onPress: (file: WorkspaceFile) => void;
}

export function FileRow({ file, onPress }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const icon = fileKind(file) === "image" ? Image01Icon : File01Icon;

  function handlePress() {
    onPress(file);
  }

  return (
    <Pressable
      accessibilityRole="button"
      onPress={handlePress}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      <Icon icon={icon} size={20} color={colors.textTertiary} />
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
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  text: { flex: 1, gap: 2 },
  name: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
}));
