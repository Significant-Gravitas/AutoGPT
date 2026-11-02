import { memo } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import type { SessionSummary } from "../../../api/chat";
import { Avatar } from "../../../design/components/Avatar";
import { Icon } from "../../../design/components/Icon";
import { colors, spacing, type } from "../../../design/theme";
import { formatListTimestamp } from "../../../lib/time";
import type { Participant } from "../../company/participants";

interface Props {
  session: SessionSummary;
  participant: Participant;
  onPress: (sessionID: string) => void;
}

export const ChatRow = memo(function ChatRow({
  session,
  participant,
  onPress,
}: Props) {
  const working =
    session.isProcessing ||
    session.chatStatus === "running" ||
    session.chatStatus === "queued";
  const title = session.title ?? "New conversation";
  const timestamp = formatListTimestamp(session.updatedAt);
  const status = working
    ? session.chatStatus === "queued"
      ? "Queued"
      : "Working…"
    : null;
  return (
    <Pressable
      onPress={() => onPress(session.id)}
      accessibilityRole="button"
      accessibilityLabel={[
        title,
        participant.name,
        status,
        timestamp,
        session.isPinned ? "Pinned" : null,
      ]
        .filter(Boolean)
        .join(", ")}
      testID={`chat-row-${session.id}`}
      style={({ pressed }) => [
        styles.row,
        pressed && { backgroundColor: colors.fill },
      ]}
    >
      <Avatar name={participant.name} uri={participant.avatarURL} size={48} />
      <View style={styles.body}>
        <View style={styles.topLine}>
          <Text style={[type.headline, styles.title]} numberOfLines={1}>
            {title}
          </Text>
          {session.isPinned ? (
            <Icon name="pin.fill" size={12} color={colors.tertiaryLabel} />
          ) : null}
          <Text style={[type.subheadline, styles.time]}>{timestamp}</Text>
        </View>
        <Text
          style={[
            type.subheadline,
            { color: working ? colors.tint : colors.secondaryLabel },
          ]}
          numberOfLines={1}
        >
          {status ? `${participant.name} · ${status}` : participant.name}
        </Text>
      </View>
    </Pressable>
  );
});

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    minHeight: 72,
  },
  body: { flex: 1, gap: 2 },
  topLine: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  title: { flex: 1, color: colors.label },
  time: { color: colors.secondaryLabel },
});
