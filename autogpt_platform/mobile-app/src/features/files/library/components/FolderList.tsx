import { ArrowRight01Icon, Folder01Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { WorkspaceFolder } from "@/lib/api/workspace";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { folderSummary } from "../../libraryHelpers";
import { SectionHeading } from "./SectionHeading";

interface Props {
  folders: WorkspaceFolder[];
  allFolders: WorkspaceFolder[];
  onOpen: (id: string) => void;
}

export function FolderList({ folders, allFolders, onOpen }: Props) {
  const styles = useStyles();
  const colors = useColors();
  if (folders.length === 0) return null;

  return (
    <View>
      <SectionHeading title="Folders" count={folders.length} />
      <View style={styles.card}>
        {folders.map((folder, index) => (
          <Pressable
            key={folder.id}
            accessibilityRole="button"
            accessibilityLabel={`${folder.name}, ${folderSummary(folder, allFolders)}`}
            onPress={() => onOpen(folder.id)}
            style={({ pressed }) => [styles.row, index > 0 && styles.divider, pressed && styles.pressed]}
          >
            <View style={styles.tile}>
              <Icon icon={Folder01Icon} size={19} color={colors.textSecondary} />
            </View>
            <View style={styles.text}>
              <Text numberOfLines={1} style={styles.name}>
                {folder.name}
              </Text>
              <Text style={styles.detail}>{folderSummary(folder, allFolders)}</Text>
            </View>
            <Icon icon={ArrowRight01Icon} size={16} color={colors.textTertiary} />
          </Pressable>
        ))}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: { marginHorizontal: 14, marginBottom: 18, borderRadius: radius.lg, backgroundColor: colors.panelCard, overflow: "hidden" },
  row: { flexDirection: "row", alignItems: "center", gap: spacing.md, paddingHorizontal: spacing.md, paddingVertical: 10 },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  pressed: { backgroundColor: colors.surfacePressed },
  tile: { width: 40, height: 40, borderRadius: 10, alignItems: "center", justifyContent: "center", backgroundColor: colors.panelBackground },
  text: { flex: 1, minWidth: 0 },
  name: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.text },
  detail: { fontSize: 13, lineHeight: 18, color: colors.textTertiary },
}));
