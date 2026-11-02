import { memo } from "react";
import { StyleSheet, Text, View } from "react-native";

import { PlainButton } from "../../../design/components/Buttons";
import { Icon } from "../../../design/components/Icon";
import { colors, radius, spacing, type } from "../../../design/theme";
import type { TimelineItem } from "../model/timeline";

interface Props {
  item: Extract<TimelineItem, { kind: "user" }>;
  canRetry: boolean;
  onRetry: (messageID: string) => void;
  onDiscard: (messageID: string) => void;
}

export const UserMessage = memo(function UserMessage({
  item,
  canRetry,
  onRetry,
  onDiscard,
}: Props) {
  const local = item.local;
  return (
    <View style={styles.container}>
      <View
        style={[styles.bubble, local?.state === "sending" && styles.sending]}
      >
        {item.text ? (
          <Text selectable style={[type.body, styles.text]}>
            {item.text}
          </Text>
        ) : null}
        {item.attachments.map((attachment) => (
          <View
            key={attachment.id}
            style={styles.attachment}
            accessible
            accessibilityLabel={`Attachment ${attachment.name}`}
          >
            <Icon
              name={attachment.kind === "folder" ? "folder" : "doc"}
              size={14}
              color={colors.onProminent}
            />
            <Text style={[type.footnote, styles.text]} numberOfLines={1}>
              {attachment.name}
            </Text>
          </View>
        ))}
      </View>
      {local?.state === "sending" ? (
        <Text style={[type.caption, styles.status]}>Sending…</Text>
      ) : null}
      {local?.state === "failed" ? (
        <View style={styles.failure} accessibilityLiveRegion="polite">
          <Text style={[type.footnote, styles.error]} accessibilityRole="alert">
            Not sent. {local.error}
          </Text>
          <View style={styles.actions}>
            <PlainButton
              title="Delete"
              onPress={() => onDiscard(local.messageID)}
            />
            <PlainButton
              title="Retry"
              onPress={() => onRetry(local.messageID)}
              disabled={!canRetry}
            />
          </View>
        </View>
      ) : null}
    </View>
  );
});

const styles = StyleSheet.create({
  container: {
    alignItems: "flex-end",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xs,
  },
  bubble: {
    maxWidth: "85%",
    backgroundColor: colors.prominent,
    borderRadius: radius.bubble,
    paddingHorizontal: spacing.md + 2,
    paddingVertical: spacing.sm + 2,
    gap: spacing.xs,
  },
  sending: { opacity: 0.6 },
  text: { color: colors.onProminent },
  attachment: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  status: { color: colors.secondaryLabel, marginTop: spacing.xs },
  failure: { alignItems: "flex-end", marginTop: spacing.xs, maxWidth: "85%" },
  error: { color: colors.danger, textAlign: "right" },
  actions: { flexDirection: "row" },
});
