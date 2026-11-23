import { LockIcon, SecurityCheckIcon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import { ProfileSection } from "./ProfileSection";

interface Props {
  /** The backend's own strings, quoted as they are. */
  rules: string[];
}

/** The rules built into every expert, one tile each, as on the web. */
export function ProtectedRules({ rules }: Props) {
  const styles = useStyles();
  const colors = useColors();
  if (rules.length === 0) return null;

  return (
    <ProfileSection
      title="Rules this expert cannot break"
      description="Part of every expert's soul, and not editable by anyone."
    >
      <View style={styles.list}>
        {rules.map((rule) => (
          <View key={rule} style={styles.rule}>
            <View style={styles.shield}>
              <Icon icon={SecurityCheckIcon} size={19} color={colors.success} />
            </View>
            <View style={styles.text}>
              <Text style={styles.ruleText}>{rule}</Text>
              <View style={styles.locked}>
                <Icon icon={LockIcon} size={12} color={colors.textTertiary} />
                <Text style={styles.lockedText}>Locked</Text>
              </View>
            </View>
          </View>
        ))}
      </View>
    </ProfileSection>
  );
}

const useStyles = createStyles((colors) => ({
  list: { gap: 10 },
  rule: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 13,
    padding: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
  },
  shield: {
    width: 38,
    height: 38,
    borderRadius: 11,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.hiredBadge,
  },
  text: { flex: 1, minWidth: 0, gap: 4 },
  ruleText: { fontSize: 15.5, lineHeight: 21, fontWeight: "500", color: colors.text },
  locked: { flexDirection: "row", alignItems: "center", gap: 4 },
  lockedText: { fontSize: 12.5, lineHeight: 16, color: colors.textTertiary },
}));
