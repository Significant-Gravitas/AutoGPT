import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { chatCountLabel, type ChatGroup } from "@/features/drawer/helpers";
import { createStyles, radius, spacing } from "@/lib/theme";

import { ChatListRow } from "./ChatListRow";

const PREVIEW_COUNT = 4;

interface Props {
  group: ChatGroup;
  isTruncated: boolean;
  onOpenChat: (sessionID: string) => void;
}

export function ChatGroupSection({ group, isTruncated, onOpenChat }: Props) {
  const styles = useStyles();
  const [isExpanded, setIsExpanded] = useState(false);
  const shown = isExpanded
    ? group.sessions
    : group.sessions.slice(0, PREVIEW_COUNT);
  const hiddenCount = group.sessions.length - shown.length;

  function showAll() {
    setIsExpanded(true);
  }

  return (
    <View style={styles.section}>
      <View style={styles.header}>
        <ExpertAvatar
          name={group.label}
          source={group.avatar}
          tint={group.tint}
          size={36}
        />
        <View style={styles.identity}>
          <Text numberOfLines={1} style={styles.name}>
            {group.label}
          </Text>
          {group.role ? (
            <Text numberOfLines={1} style={styles.role}>
              {group.role}
            </Text>
          ) : null}
        </View>
        <Text style={styles.count}>
          {chatCountLabel(group.sessions.length, isTruncated)}
        </Text>
      </View>
      <View style={styles.card}>
        {shown.map((session, index) => (
          <ChatListRow
            key={session.id}
            session={session}
            isFirst={index === 0}
            onOpenChat={onOpenChat}
          />
        ))}
        {hiddenCount > 0 ? (
          <Pressable
            accessibilityRole="button"
            onPress={showAll}
            style={({ pressed }) => [styles.more, pressed && styles.pressed]}
          >
            <Text style={styles.moreLabel}>Show {hiddenCount} more</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.sm },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.xs,
  },
  identity: { flex: 1, minWidth: 0 },
  name: { fontSize: 15, fontWeight: "600", lineHeight: 20, color: colors.text },
  role: { fontSize: 12, lineHeight: 16, color: colors.textTertiary },
  count: {
    fontSize: 12,
    lineHeight: 16,
    color: colors.textSecondary,
    backgroundColor: colors.surfaceRaised,
    borderRadius: radius.pill,
    overflow: "hidden",
    paddingHorizontal: 6,
  },
  card: {
    backgroundColor: colors.surface,
    borderWidth: 0.5,
    borderColor: colors.border,
    borderRadius: 16,
    overflow: "hidden",
  },
  more: {
    minHeight: 44,
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  moreLabel: {
    fontSize: 13,
    fontWeight: "500",
    lineHeight: 18,
    color: colors.accentText,
  },
}));
