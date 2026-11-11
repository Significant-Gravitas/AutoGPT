import { AlertCircleIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { DynamicToolUIPart, ToolUIPart } from "ai";
import { ActivityIndicator, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  part: ToolUIPart | DynamicToolUIPart;
}

export function ToolRow({ part }: Props) {
  const styles = useStyles();
  const title =
    typeof part.title === "string" && part.title.trim()
      ? part.title
      : "Working on it";
  const label =
    part.state === "output-error" ? `${title} · didn't finish` : title;

  return (
    <View style={styles.row}>
      <StateMark state={part.state} />
      <Text numberOfLines={1} style={styles.label}>
        {label}
      </Text>
    </View>
  );
}

interface StateMarkProps {
  state: Props["part"]["state"];
}

function StateMark({ state }: StateMarkProps) {
  const colors = useColors();
  if (state === "input-streaming" || state === "input-available") {
    return <ActivityIndicator size="small" color={colors.textSecondary} />;
  }
  if (state === "output-available") {
    return <Icon icon={Tick02Icon} size={16} color={colors.success} />;
  }
  if (state === "output-error") {
    return <Icon icon={AlertCircleIcon} size={16} color={colors.error} />;
  }
  return null;
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    maxWidth: "100%",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
  },
  label: { ...type.small, flexShrink: 1, color: colors.textSecondary },
}));
