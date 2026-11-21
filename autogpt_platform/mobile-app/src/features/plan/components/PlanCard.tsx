import { StyleSheet, Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import { clampPercent, type BadgeTone, type Meter, type PlanSummary } from "../helpers";

interface Props {
  summary: PlanSummary;
  meters: Meter[];
}

/** What the account is on, when money next moves, and how much is used. Read only. */
export function PlanCard({ summary, meters }: Props) {
  const styles = useStyles();
  const badgeStyles: Record<BadgeTone, object> = {
    active: styles.badgeActive,
    trial: styles.badgeTrial,
    ending: styles.badgeEnding,
  };
  const badgeTextStyles: Record<BadgeTone, object> = {
    active: styles.badgeActiveText,
    trial: styles.badgeTrialText,
    ending: styles.badgeEndingText,
  };
  const { badge } = summary;

  return (
    <View style={styles.card}>
      <View style={styles.titleRow}>
        <Text style={styles.name}>{summary.name}</Text>
        <View style={[styles.badge, badgeStyles[badge.tone]]}>
          <Text style={[styles.badgeText, badgeTextStyles[badge.tone]]}>
            {badge.label}
          </Text>
        </View>
      </View>
      {summary.headline || summary.detail ? (
        <View>
          {summary.headline ? <Text style={styles.headline}>{summary.headline}</Text> : null}
          {summary.detail ? <Text style={styles.detail}>{summary.detail}</Text> : null}
        </View>
      ) : null}
      {meters.map((meter) => (
        <View key={meter.label} style={styles.meter}>
          <View style={styles.meterTop}>
            <Text style={styles.meterLabel}>{meter.label}</Text>
            <Text style={styles.meterNote}>{meter.note}</Text>
          </View>
          <View
            accessibilityRole="progressbar"
            accessibilityLabel={meter.label}
            accessibilityValue={{ min: 0, max: 100, now: Math.round(clampPercent(meter.percent)) }}
            style={styles.track}
          >
            <View style={[styles.fill, { width: `${clampPercent(meter.percent)}%` }]} />
          </View>
        </View>
      ))}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    gap: spacing.md,
    padding: spacing.lg,
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.panelCard,
  },
  titleRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.headline, color: colors.text },
  badge: { paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: radius.pill },
  badgeText: { ...type.small, fontWeight: "600" },
  badgeActive: { backgroundColor: colors.hiredBadge },
  badgeActiveText: { color: colors.onHiredBadge },
  badgeTrial: { backgroundColor: colors.brandSoft },
  badgeTrialText: { color: colors.accentText },
  badgeEnding: { backgroundColor: colors.surfaceRaised },
  badgeEndingText: { color: colors.textSecondary },
  headline: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textSecondary },
  meter: { gap: 6 },
  meterTop: { flexDirection: "row", justifyContent: "space-between" },
  meterLabel: { ...type.small, color: colors.text },
  meterNote: { ...type.small, color: colors.textTertiary },
  track: { height: 6, overflow: "hidden", borderRadius: radius.pill, backgroundColor: colors.surfaceRaised },
  fill: { height: "100%", borderRadius: radius.pill, backgroundColor: colors.text },
}));
