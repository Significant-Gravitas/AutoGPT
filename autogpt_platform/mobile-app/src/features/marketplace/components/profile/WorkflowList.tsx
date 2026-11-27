import { Calendar03Icon, FlashIcon } from "@hugeicons/core-free-icons";
import { StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { ExpertWorkflow } from "@/lib/api/experts";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { ProfileSection } from "./ProfileSection";

interface Props {
  name: string;
  workflows: ExpertWorkflow[];
}

/** A quiet list at the foot of the page: each workflow's name and one line on it. */
export function WorkflowList({ name, workflows }: Props) {
  const styles = useStyles();
  const colors = useColors();
  if (workflows.length === 0) return null;

  return (
    <ProfileSection
      title="Workflows"
      count={workflows.length}
      description={`Installed and ready to run the moment ${name} joins your team.`}
    >
      <View style={styles.card}>
        {workflows.map((workflow, index) => (
          <View key={workflow.id} style={[styles.row, index > 0 && styles.divider]}>
            <Icon icon={FlashIcon} size={17} color={colors.textTertiary} />
            <View style={styles.text}>
              <Text numberOfLines={1} style={styles.title}>
                {workflow.name ?? "Unnamed workflow"}
              </Text>
              {workflow.description ? (
                <Text numberOfLines={1} style={styles.description}>
                  {workflow.description}
                </Text>
              ) : null}
              {workflow.schedule_cron ? (
                <View style={styles.schedule}>
                  <Icon icon={Calendar03Icon} size={12} color={colors.textTertiary} />
                  <Text style={styles.description}>Runs on a schedule</Text>
                </View>
              ) : null}
            </View>
          </View>
        ))}
      </View>
    </ProfileSection>
  );
}

const useStyles = createStyles((colors) => ({
  card: { borderRadius: radius.lg, backgroundColor: colors.panelCard, overflow: "hidden" },
  row: { flexDirection: "row", alignItems: "flex-start", gap: spacing.md, padding: 14 },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  text: { flex: 1, minWidth: 0, gap: 1 },
  title: { fontSize: 15, lineHeight: 19, fontWeight: "600", color: colors.text },
  description: { fontSize: 13.5, lineHeight: 18, color: colors.textTertiary },
  schedule: { flexDirection: "row", alignItems: "center", gap: 4, marginTop: 2 },
}));
