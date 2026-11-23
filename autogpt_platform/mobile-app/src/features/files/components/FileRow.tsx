import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";

import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { fileDetailLine } from "../listHelpers";
import { FileThumbnail } from "./FileThumbnail";

/** Who made a file, as the Files page names them. */
export interface FileOwner {
  name: string;
  avatar: string | null;
  tint: string | null;
}

interface Props {
  file: WorkspaceFile;
  isFirst: boolean;
  owner?: FileOwner | null;
  onPress: (file: WorkspaceFile) => void;
}

export function FileRow({ file, isFirst, owner, onPress }: Props) {
  // "Today" and "Yesterday" move with the clock, so the row is never cached.
  "use no memo";
  const styles = useStyles();
  const colors = useColors();

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
        <View style={styles.detailRow}>
          {owner?.avatar ? (
            <ExpertAvatar name={owner.name} source={owner.avatar} tint={owner.tint} size={16} />
          ) : null}
          <Text numberOfLines={1} style={styles.detail}>
            {owner ? <Text style={styles.owner}>{`${owner.name} · `}</Text> : null}
            {fileDetailLine(file)}
          </Text>
        </View>
      </View>
      <Icon icon={ArrowRight01Icon} size={16} color={colors.textTertiary} />
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
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  pressed: { backgroundColor: colors.surfacePressed },
  text: { flex: 1, minWidth: 0 },
  name: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.text },
  detailRow: { flexDirection: "row", alignItems: "center", gap: 5 },
  detail: { flexShrink: 1, fontSize: 13, lineHeight: 18, color: colors.textTertiary },
  owner: { fontWeight: "500", color: colors.textSecondary },
}));
