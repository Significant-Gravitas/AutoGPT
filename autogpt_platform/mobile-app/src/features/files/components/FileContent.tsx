import { Image } from "expo-image";
import {
  ActivityIndicator,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Markdown } from "@/features/chat/components/Markdown";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { colors, spacing, type } from "@/lib/theme";

import { useFileViewer } from "../useFileViewer";
import { RetryNotice } from "./RetryNotice";

interface Props {
  file: WorkspaceFile;
}

export function FileContent({ file }: Props) {
  const viewer = useFileViewer(file);
  const insets = useSafeAreaInsets();

  if (viewer.kind === "other") {
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>
          {"This file can't be opened here yet. Open it on the web."}
        </Text>
      </View>
    );
  }
  if (viewer.isError) {
    return (
      <View style={styles.center}>
        <RetryNotice
          centered
          message="Couldn't open this file. Try again."
          onRetry={viewer.retry}
        />
      </View>
    );
  }
  if (viewer.kind === "image" && viewer.image) {
    return (
      <Image
        key={viewer.imageAttempt}
        source={viewer.image}
        contentFit="contain"
        accessibilityLabel={file.name}
        onError={viewer.handleImageError}
        style={[styles.image, { marginBottom: insets.bottom }]}
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

  return (
    <ScrollView
      contentContainerStyle={[
        styles.content,
        { paddingBottom: insets.bottom + spacing.xl },
      ]}
    >
      {viewer.isCapped ? (
        <Text style={styles.capped}>Showing the first 200 KB.</Text>
      ) : null}
      {viewer.kind === "markdown" ? (
        <Markdown text={viewer.text} />
      ) : (
        <Text selectable style={styles.code}>
          {viewer.text}
        </Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  image: { flex: 1 },
  content: { padding: spacing.lg, gap: spacing.md },
  capped: { ...type.small, color: colors.textTertiary },
  code: {
    ...type.small,
    color: colors.text,
    fontFamily: Platform.select({ ios: "Menlo", default: "monospace" }),
  },
});
