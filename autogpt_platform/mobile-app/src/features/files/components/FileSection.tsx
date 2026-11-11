import { ActivityIndicator, Text, View } from "react-native";

import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { FileRow } from "./FileRow";
import { RetryNotice } from "./RetryNotice";

interface Props {
  title: string;
  emptyText: string;
  files: WorkspaceFile[];
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
  onSelect: (file: WorkspaceFile) => void;
}

export function FileSection({
  title,
  emptyText,
  files,
  isLoading,
  isError,
  onRetry,
  onSelect,
}: Props) {
  const styles = useStyles();
  const colors = useColors();

  function renderBody() {
    if (isError) {
      return <RetryNotice message="Couldn't load files. Try again." onRetry={onRetry} />;
    }
    if (isLoading) {
      return <ActivityIndicator color={colors.textTertiary} style={styles.spinner} />;
    }
    if (files.length === 0) return <Text style={styles.empty}>{emptyText}</Text>;
    return files.map((file) => <FileRow key={file.id} file={file} onPress={onSelect} />);
  }

  return (
    <View style={styles.section}>
      <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
        {title}
      </Text>
      <View style={styles.body}>{renderBody()}</View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.sm },
  title: {
    ...type.small,
    fontWeight: "600",
    color: colors.textSecondary,
    paddingHorizontal: spacing.md,
  },
  body: { paddingHorizontal: spacing.xs },
  spinner: { alignSelf: "flex-start", marginHorizontal: spacing.md },
  empty: {
    ...type.small,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
  },
}));
