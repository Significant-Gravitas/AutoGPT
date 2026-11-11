import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { chatCountLabel, type ChatGroup } from "@/features/drawer/helpers";
import type { SessionSummary } from "@/lib/api/chat";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { DrawerRow } from "../DrawerRow";

// Otto opens with a short preview and a single "Load all"; every other
// expert starts collapsed and pages in when opened, as on the web.
const OTTO_PREVIEW_COUNT = 1;
const EXPERT_PAGE_SIZE = 4;
const AVATAR_SIZE = 32;

interface Props {
  group: ChatGroup;
  isTruncated: boolean;
  activeSessionID?: string;
  onOpenChat: (sessionID: string) => void;
}

export function ExpertChatGroup({
  group,
  isTruncated,
  activeSessionID,
  onOpenChat,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const [isOpen, setIsOpen] = useState(group.isOtto);
  const [visibleCount, setVisibleCount] = useState(
    group.isOtto ? OTTO_PREVIEW_COUNT : EXPERT_PAGE_SIZE,
  );
  const shown = isOpen
    ? group.sessions.slice(0, visibleCount)
    : group.sessions.filter((session) => session.is_processing);
  const hasMore = isOpen && group.sessions.length > shown.length;

  function toggle() {
    setIsOpen((open) => !open);
  }

  function showMore() {
    setVisibleCount((count) =>
      group.isOtto ? Infinity : count + EXPERT_PAGE_SIZE,
    );
  }

  function renderSession(session: SessionSummary) {
    return (
      <DrawerRow
        key={session.id}
        label={session.title || "New chat"}
        isActive={session.id === activeSessionID}
        isBusy={session.is_processing}
        onPress={() => onOpenChat(session.id)}
      />
    );
  }

  return (
    <View>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`${group.label} chats`}
        accessibilityState={{ expanded: isOpen }}
        onPress={toggle}
        style={({ pressed }) => [styles.header, pressed && styles.pressed]}
      >
        <ExpertAvatar
          name={group.label}
          source={group.avatar}
          tint={group.tint}
          size={AVATAR_SIZE}
        />
        <View style={styles.identity}>
          <View style={styles.nameRow}>
            <Text numberOfLines={1} style={styles.name}>
              {group.label}
            </Text>
            <Text style={styles.count}>
              {chatCountLabel(group.sessions.length, isTruncated)}
            </Text>
          </View>
          {group.role ? (
            <Text numberOfLines={1} style={styles.role}>
              {group.role}
            </Text>
          ) : null}
        </View>
        <View style={isOpen ? styles.chevronOpen : undefined}>
          <Icon icon={ArrowDown01Icon} size={18} color={colors.textTertiary} />
        </View>
      </Pressable>
      {shown.length > 0 || hasMore ? (
        <View style={styles.body}>
          {shown.map(renderSession)}
          {hasMore ? (
            <Pressable
              accessibilityRole="button"
              onPress={showMore}
              style={({ pressed }) => [styles.more, pressed && styles.pressed]}
            >
              <Text style={styles.moreLabel}>
                {group.isOtto ? "Load all" : "Load more"}
              </Text>
            </Pressable>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  identity: { flex: 1, minWidth: 0 },
  nameRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.label, color: colors.text, flexShrink: 1 },
  count: {
    ...type.small,
    fontSize: 12,
    lineHeight: 16,
    color: colors.textSecondary,
    backgroundColor: colors.surfaceRaised,
    borderRadius: radius.pill,
    overflow: "hidden",
    paddingHorizontal: 6,
  },
  role: { ...type.small, color: colors.textTertiary },
  chevronOpen: { transform: [{ rotate: "180deg" }] },
  body: {
    marginLeft: spacing.md + AVATAR_SIZE / 2,
    paddingLeft: spacing.xs,
    borderLeftWidth: StyleSheet.hairlineWidth,
    borderLeftColor: colors.border,
    gap: 2,
  },
  more: {
    minHeight: 36,
    justifyContent: "center",
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  moreLabel: { ...type.small, color: colors.textTertiary },
}));
