import { Clock01Icon, Notification01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { Pressable, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { promptTopic } from "@/features/chat/promptTopic";
import { avatarSource } from "@/features/team/helpers";
import type { HomeExpert } from "@/lib/api/home";
import { expertTint } from "@/lib/expertPalette";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { nextRunLabel, type HomeRow } from "../helpers";
import { openHomeLink } from "../openHomeLink";

const LEAD_SIZE = 26;

interface Props {
  rows: HomeRow[];
  onPick: (prompt: string) => void;
}

export function HomeRows({ rows, onPick }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.list}>
      {rows.map((row) => (
        <HomeRowItem key={rowKey(row)} row={row} onPick={onPick} />
      ))}
    </View>
  );
}

function rowKey(row: HomeRow) {
  if (row.kind === "attention") return `attention-${row.item.id}`;
  if (row.kind === "prompt") return `prompt-${row.prompt}`;
  return `${row.kind}-${row.task.id}`;
}

interface ItemProps {
  row: HomeRow;
  onPick: (prompt: string) => void;
}

function HomeRowItem({ row, onPick }: ItemProps) {
  const styles = useStyles();
  const colors = useColors();
  const muted = colors.textSecondary;

  if (row.kind === "attention") {
    const { item } = row;
    return (
      <Row
        lead={
          item.expert ? (
            <Avatar expert={item.expert} />
          ) : (
            <Icon icon={Notification01Icon} size={22} color={muted} />
          )
        }
        label={`${item.title}, ${item.primary_action.label}`}
        onPress={() => openHomeLink(item.primary_action.href)}
      >
        {item.title}
      </Row>
    );
  }
  if (row.kind === "active") {
    const { task } = row;
    return (
      <Row
        lead={
          task.expert ? (
            <Avatar expert={task.expert} />
          ) : (
            <Icon icon={Clock01Icon} size={22} color={muted} />
          )
        }
        label={`${task.title}, working now`}
        trailing={<View style={styles.live} />}
      >
        {task.title}
      </Row>
    );
  }
  if (row.kind === "upcoming") {
    const { task } = row;
    const when = nextRunLabel(task.next_run_time);
    return (
      <Row
        lead={<Icon icon={Clock01Icon} size={22} color={muted} />}
        label={`${task.title}, next up ${when}`}
      >
        {task.title}
        <Text style={styles.detail}>{` · ${when}`}</Text>
      </Row>
    );
  }
  const { prompt } = row;
  return (
    <Row
      lead={<Icon icon={promptTopic(prompt).icon} size={22} color={muted} />}
      label={prompt}
      onPress={() => onPick(prompt)}
    >
      {prompt}
    </Row>
  );
}

interface RowProps {
  lead: ReactNode;
  label: string;
  children: ReactNode;
  trailing?: ReactNode;
  onPress?: () => void;
}

function Row({ lead, label, children, trailing, onPress }: RowProps) {
  const styles = useStyles();
  const body = (
    <>
      <View style={styles.lead}>{lead}</View>
      <Text numberOfLines={2} style={styles.text}>
        {children}
      </Text>
      {trailing}
    </>
  );
  if (!onPress) {
    return (
      <View accessible accessibilityLabel={label} style={styles.row}>
        {body}
      </View>
    );
  }
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      {body}
    </Pressable>
  );
}

function Avatar({ expert }: { expert: HomeExpert }) {
  return (
    <ExpertAvatar
      name={expert.name}
      source={avatarSource(expert)}
      size={LEAD_SIZE}
      tint={expertTint({ avatarUrl: expert.avatar_url, role: expert.role })}
    />
  );
}

const useStyles = createStyles((colors) => ({
  list: { gap: 2 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    paddingVertical: 9,
  },
  pressed: { opacity: 0.5 },
  lead: { width: 28, alignItems: "center" },
  text: { flex: 1, fontSize: 16, lineHeight: 22, color: colors.text },
  detail: { color: colors.textTertiary },
  live: {
    width: 7,
    height: 7,
    marginRight: spacing.xs,
    borderRadius: 4,
    backgroundColor: colors.success,
  },
}));
