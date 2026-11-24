import { ArrowRight01Icon, Folder01Icon } from "@hugeicons/core-free-icons";
import { Fragment } from "react";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { StorageUsage, WorkspaceFolder } from "@/lib/api/workspace";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { folderSummary, storageLine } from "../../libraryHelpers";

interface Props {
  path: WorkspaceFolder[];
  allFolders: WorkspaceFolder[];
  storage: StorageUsage | null;
  onOpenFolder: (id: string | null) => void;
}

/** The top level's title and storage, or a folder's path and name. */
export function PageTop({ path, allFolders, storage, onOpenFolder }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const folder = path[path.length - 1];

  if (!folder) {
    const share = storage && storage.limit_bytes > 0 ? Math.min(1, storage.used_bytes / storage.limit_bytes) : 0;
    return (
      <View style={styles.top}>
        <Text accessibilityRole="header" style={styles.title}>
          Files
        </Text>
        <Text style={styles.sub}>Every file your team makes or you upload.</Text>
        {storage ? (
          <View accessibilityLabel={`Storage: ${storageLine(storage)} used`} style={styles.storage}>
            <Text style={styles.storageText}>{storageLine(storage)}</Text>
            <View style={styles.track}>
              <View style={[styles.fill, { width: `${Math.max(share * 100, 1)}%` }]} />
            </View>
          </View>
        ) : null}
      </View>
    );
  }

  return (
    <View style={styles.top}>
      <View style={styles.crumbs}>
        <Pressable accessibilityRole="link" onPress={() => onOpenFolder(null)} hitSlop={6}>
          <Text style={styles.crumb}>Files</Text>
        </Pressable>
        {path.map((step) => (
          <Fragment key={step.id}>
            <Icon icon={ArrowRight01Icon} size={12} color={colors.textTertiary} />
            {step.id === folder.id ? (
              <Text numberOfLines={1} style={[styles.crumb, styles.current]}>
                {step.name}
              </Text>
            ) : (
              <Pressable accessibilityRole="link" onPress={() => onOpenFolder(step.id)} hitSlop={6}>
                <Text numberOfLines={1} style={styles.crumb}>
                  {step.name}
                </Text>
              </Pressable>
            )}
          </Fragment>
        ))}
      </View>
      <View style={styles.folderTitle}>
        <View style={styles.folderTile}>
          <Icon icon={Folder01Icon} size={22} color={colors.textSecondary} />
        </View>
        <View style={styles.folderText}>
          <Text accessibilityRole="header" numberOfLines={2} style={styles.title}>
            {folder.name}
          </Text>
          <Text style={styles.storageText}>{folderSummary(folder, allFolders)}</Text>
        </View>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  top: { paddingHorizontal: 18, paddingTop: 10, paddingBottom: 14 },
  title: { fontSize: 23, lineHeight: 28, fontWeight: "700", letterSpacing: -0.4, color: colors.text },
  sub: { marginTop: 2, fontSize: 13, lineHeight: 18, color: colors.textSecondary },
  storage: { marginTop: 10, flexDirection: "row", alignItems: "center", gap: 10 },
  storageText: { fontSize: 12, lineHeight: 16, color: colors.textTertiary },
  track: { flex: 1, height: 4, borderRadius: 2, backgroundColor: colors.panelTabTrack, overflow: "hidden" },
  fill: { height: 4, borderRadius: 2, backgroundColor: colors.textSecondary },
  crumbs: { flexDirection: "row", alignItems: "center", gap: 4, marginBottom: 10, flexWrap: "wrap" },
  crumb: { fontSize: 13, lineHeight: 18, color: colors.textTertiary, maxWidth: 160 },
  current: { color: colors.text, fontWeight: "600" },
  folderTitle: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  folderTile: {
    width: 44,
    height: 44,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.panelCard,
  },
  folderText: { flex: 1, minWidth: 0 },
}));
