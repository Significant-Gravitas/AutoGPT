import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import type { ScheduleInfo } from "@/lib/api/expertActivity";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { nextRunText } from "../helpers";
import { ActivityRow } from "./ActivityRow";
import { ActivitySection } from "./ActivitySection";

interface Props {
  isLoading: boolean;
  schedules: ScheduleInfo[];
  hiddenCount: number;
  onShowMore: () => void;
  isPaused: boolean;
  onResume: () => void;
  isResuming: boolean;
  resumeError: string | null;
}

export function UpcomingSection({
  isLoading,
  schedules,
  hiddenCount,
  onShowMore,
  isPaused,
  onResume,
  isResuming,
  resumeError,
}: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <ActivitySection title="Upcoming" isLoading={isLoading}>
      {isPaused ? (
        <View>
          <ActivityRow
            title="Routines paused"
            trailing={
              <Button
                label="Resume"
                onPress={onResume}
                variant="secondary"
                size="small"
                isLoading={isResuming}
              />
            }
          />
          {resumeError ? <Text style={styles.error}>{resumeError}</Text> : null}
        </View>
      ) : null}
      {schedules.map((schedule) => (
        <ActivityRow
          key={schedule.id}
          leading={<Icon icon={Clock01Icon} size={20} color={colors.textTertiary} />}
          title={schedule.name}
          detail={nextRunText(schedule.next_run_time)}
        />
      ))}
      {hiddenCount > 0 ? (
        <Pressable
          accessibilityRole="button"
          onPress={onShowMore}
          style={({ pressed }) => [styles.more, pressed && styles.pressed]}
        >
          <Text style={styles.moreLabel}>Show more</Text>
        </Pressable>
      ) : null}
    </ActivitySection>
  );
}

const useStyles = createStyles((colors) => ({
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
  },
  more: { alignItems: "center", paddingVertical: spacing.md },
  pressed: { backgroundColor: colors.surfacePressed },
  moreLabel: { ...type.small, color: colors.textSecondary },
}));
