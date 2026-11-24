import { ActivityIndicator, Platform, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Markdown } from "@/features/chat/components/Markdown";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { formatJson, tableFromText } from "../libraryHelpers";
import { useFileViewer } from "../useFileViewer";
import { RetryNotice } from "./RetryNotice";
import { CodeView } from "./viewers/CodeView";
import { ImageView } from "./viewers/ImageView";
import { OtherFileView } from "./viewers/OtherFileView";
import { PdfView } from "./viewers/PdfView";
import { TableView } from "./viewers/TableView";

interface Props {
  file: WorkspaceFile;
  isSharing: boolean;
  onOpenIn: () => void;
}

// Only iOS's web view draws a PDF; Android's shows a blank page.
const CAN_SHOW_PDF = Platform.OS === "ios";

export function FileContent({ file, isSharing, onOpenIn }: Props) {
  const viewer = useFileViewer(file);
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();

  if (viewer.kind === "other" || (viewer.kind === "pdf" && !CAN_SHOW_PDF)) {
    return <OtherFileView file={file} isSharing={isSharing} onOpenIn={onOpenIn} />;
  }
  if (viewer.isTooLarge) {
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>This file is too large to preview here.</Text>
      </View>
    );
  }
  if (viewer.isError) {
    return (
      <View style={styles.center}>
        <RetryNotice centered message="Couldn't open this file. Try again." onRetry={viewer.retry} />
      </View>
    );
  }
  if (viewer.kind === "image" && viewer.image) {
    return (
      <ImageView
        key={viewer.imageAttempt}
        source={viewer.image}
        label={file.name}
        onError={viewer.handleImageError}
      />
    );
  }
  if (viewer.kind === "pdf" && viewer.headers) {
    return (
      <PdfView
        key={viewer.imageAttempt}
        uri={viewer.downloadURL}
        headers={viewer.headers}
        onError={viewer.handleImageError}
      />
    );
  }
  if (viewer.isLoading || viewer.text === null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.textTertiary} />
      </View>
    );
  }
  if (viewer.kind === "table") {
    return <TableView table={tableFromText(viewer.text, file)} cappedNotice={viewer.cappedNotice} />;
  }
  if (viewer.kind === "text") {
    const text = /\.json$/i.test(file.name) ? formatJson(viewer.text) : viewer.text;
    return <CodeView text={text} cappedNotice={viewer.cappedNotice} />;
  }

  return (
    <ScrollView contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + spacing.xl }]}>
      {viewer.cappedNotice ? <Text style={styles.capped}>{viewer.cappedNotice}</Text> : null}
      <Markdown text={viewer.text} />
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.xl },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  content: { padding: spacing.lg, gap: spacing.md },
  capped: { ...type.small, color: colors.textTertiary },
}));
