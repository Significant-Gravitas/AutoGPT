import { File02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { FileRow } from "./FileRow";
import { RetryNotice } from "./RetryNotice";

interface Props {
  title: string;
  /** Shown beside the title, as the web's "Files in this chat (3)". */
  showCount?: boolean;
  emptyText?: string;
  files: WorkspaceFile[];
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
  onSelect: (file: WorkspaceFile) => void;
}

const COLLAPSED_FILE_COUNT = 4;

export function FileSection({
  title,
  showCount,
  emptyText,
  files,
  isLoading,
  isError,
  onRetry,
  onSelect,
}: Props) {
  const [showAll, setShowAll] = useState(false);
  const styles = useStyles();
  const colors = useColors();
  const count = showCount && files.length > 0 ? ` (${files.length})` : "";

  function toggleShowAll() {
    setShowAll(!showAll);
  }

  function renderBody() {
    if (isError) return <RetryNotice message="Failed to load files." onRetry={onRetry} />;
    if (isLoading) {
      return (
        <View style={styles.skeleton}>
          <View style={[styles.bar, styles.fullBar]} />
          <View style={[styles.bar, styles.shortBar]} />
        </View>
      );
    }
    if (files.length === 0) {
      return emptyText ? <Text style={styles.empty}>{emptyText}</Text> : null;
    }
    const visible = showAll ? files : files.slice(0, COLLAPSED_FILE_COUNT);
    const hiddenCount = files.length - visible.length;
    return (
      <>
        <View>
          {visible.map((file, index) => (
            <FileRow key={file.id} file={file} isFirst={index === 0} onPress={onSelect} />
          ))}
        </View>
        {files.length > COLLAPSED_FILE_COUNT ? (
          <Pressable accessibilityRole="button" onPress={toggleShowAll} hitSlop={8}>
            <Text style={styles.more}>
              {showAll ? "Show less" : `View more (${hiddenCount})`}
            </Text>
          </Pressable>
        ) : null}
      </>
    );
  }

  return (
    <View>
      <View style={styles.header}>
        <Icon icon={File02Icon} size={14} color={colors.textTertiary} />
        <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
          {title}
          {count}
        </Text>
      </View>
      {renderBody()}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: spacing.md,
    marginBottom: 6,
  },
  title: {
    flexShrink: 1,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: "500",
    color: colors.textSecondary,
  },
  skeleton: { gap: 10, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  bar: { height: 20, borderRadius: 8, backgroundColor: colors.surfaceRaised },
  fullBar: { width: "100%" },
  shortBar: { width: "66%" },
  empty: {
    fontSize: 13,
    lineHeight: 18,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  more: {
    fontSize: 13,
    lineHeight: 18,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
    paddingTop: 6,
    paddingBottom: 2,
  },
}));
