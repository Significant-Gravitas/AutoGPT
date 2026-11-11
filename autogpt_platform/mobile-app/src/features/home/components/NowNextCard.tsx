import { Clock01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource } from "@/features/team/helpers";
import type { HomeActiveTask, HomeUpcomingTask } from "@/lib/api/home";
import { expertTint } from "@/lib/expertPalette";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { nextRunLabel } from "../helpers";

interface Props {
  active: HomeActiveTask | null;
  upcoming: HomeUpcomingTask | null;
}

export function NowNextCard({ active, upcoming }: Props) {
  const styles = useStyles();
  const colors = useColors();
  if (!active && !upcoming) return null;
  const clock = <Icon icon={Clock01Icon} size={20} color={colors.textTertiary} />;

  return (
    <View style={styles.card}>
      {active ? (
        <Row
          lead={
            active.expert ? (
              <ExpertAvatar
                name={active.expert.name}
                source={avatarSource(active.expert)}
                size={32}
                tint={expertTint({
                  avatarUrl: active.expert.avatar_url,
                  role: active.expert.role,
                })}
              />
            ) : (
              clock
            )
          }
          title={active.title}
          detail={
            active.expert ? `${active.expert.name} · working now` : "Working now"
          }
          trailing={<View style={styles.dot} />}
        />
      ) : null}
      {active && upcoming ? <View style={styles.separator} /> : null}
      {upcoming ? (
        <Row
          lead={clock}
          title={upcoming.title}
          detail={`Next up · ${nextRunLabel(upcoming.next_run_time)}`}
        />
      ) : null}
    </View>
  );
}

interface RowProps {
  lead: ReactNode;
  title: string;
  detail: string;
  trailing?: ReactNode;
}

function Row({ lead, title, detail, trailing }: RowProps) {
  const styles = useStyles();
  return (
    <View style={styles.row}>
      <View style={styles.lead}>{lead}</View>
      <View style={styles.text}>
        <Text numberOfLines={1} style={styles.title}>
          {title}
        </Text>
        <Text numberOfLines={1} style={styles.detail}>
          {detail}
        </Text>
      </View>
      {trailing}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    borderRadius: 16,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    overflow: "hidden",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  lead: { width: 32, alignItems: "center" },
  text: { flex: 1, gap: 2 },
  title: { fontSize: 15, fontWeight: "500", lineHeight: 20, color: colors.text },
  detail: { fontSize: 13, lineHeight: 18, color: colors.textTertiary },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.success },
  separator: { height: 0.5, backgroundColor: colors.border },
}));
