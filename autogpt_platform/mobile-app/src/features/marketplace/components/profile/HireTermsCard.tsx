import { CheckmarkCircle02Icon, Coins01Icon } from "@hugeicons/core-free-icons";
import { StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { hirePerks, planLine } from "../../profileHelpers";

interface Props {
  name: string;
  workflowCount: number;
  weeklyBudget: number | null | undefined;
}

/** What comes with the hire and what it costs. The pinned pill below is its button. */
export function HireTermsCard({ name, workflowCount, weeklyBudget }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <View style={styles.card}>
      <View style={styles.perks}>
        {hirePerks(workflowCount).map((perk) => (
          <View key={perk} style={styles.perk}>
            <Icon icon={CheckmarkCircle02Icon} size={18} color={colors.success} />
            <Text style={styles.perkText}>{perk}</Text>
          </View>
        ))}
      </View>
      <View style={styles.plan}>
        <View style={styles.planTitle}>
          <Icon icon={Coins01Icon} size={17} color={colors.textTertiary} />
          <Text accessibilityRole="header" style={styles.planHeading}>
            Included with your plan
          </Text>
        </View>
        <Text style={styles.planText}>{planLine(name, weeklyBudget)}</Text>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: { borderRadius: radius.lg, backgroundColor: colors.panelCard, overflow: "hidden" },
  perks: { padding: spacing.lg, gap: 11 },
  perk: { flexDirection: "row", alignItems: "flex-start", gap: 10 },
  perkText: { flex: 1, fontSize: 15, lineHeight: 20, color: colors.text },
  plan: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.panelDivider,
    padding: spacing.lg,
    gap: 6,
  },
  planTitle: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  planHeading: { fontSize: 15, lineHeight: 20, fontWeight: "600", color: colors.text },
  planText: { fontSize: 14, lineHeight: 20, color: colors.textSecondary },
}));
