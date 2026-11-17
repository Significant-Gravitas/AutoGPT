import { AlertCircleIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { stepIcons, type Step } from "../steps";

interface Props {
  step: Step;
}

export function StepRow({ step }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <View style={styles.row}>
      <View style={[styles.tile, step.state === "running" && styles.liveTile]}>
        <TileMark step={step} />
      </View>
      <Text
        numberOfLines={1}
        style={[
          styles.label,
          step.state === "running" && styles.liveLabel,
          step.state === "failed" && styles.failedLabel,
        ]}
      >
        {step.label}
      </Text>
      {step.state === "done" ? (
        <Icon icon={Tick02Icon} size={14} color={colors.textTertiary} />
      ) : null}
    </View>
  );
}

function TileMark({ step }: Props) {
  const colors = useColors();
  if (step.state === "running") {
    return <ActivityIndicator size="small" color={colors.accentText} />;
  }
  if (step.state === "failed") {
    return <Icon icon={AlertCircleIcon} size={14} color={colors.error} />;
  }
  return (
    <Icon icon={stepIcons[step.kind]} size={14} color={colors.textSecondary} />
  );
}

const useStyles = createStyles((colors) => ({
  row: { flexDirection: "row", alignItems: "center", gap: spacing.sm + 2 },
  tile: {
    width: 24,
    height: 24,
    borderRadius: 7,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.surfaceRaised,
  },
  liveTile: { backgroundColor: colors.brandSoft },
  label: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textTertiary,
  },
  liveLabel: { color: colors.text, fontWeight: "500" },
  failedLabel: { color: colors.textSecondary },
}));
