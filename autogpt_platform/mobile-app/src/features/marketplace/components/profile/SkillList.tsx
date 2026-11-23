import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { BundledSkill } from "@/lib/api/experts";
import { createStyles, radius, spacing, useScheme } from "@/lib/theme";

import { accentFill, accentInk } from "../../areaColors";
import { ProfileSection } from "./ProfileSection";

interface Props {
  skills: BundledSkill[];
  /** The expert's palette hex, for the tiles. */
  accent: string;
}

export function SkillList({ skills, accent }: Props) {
  const styles = useStyles();
  const scheme = useScheme();
  if (skills.length === 0) return null;

  return (
    <ProfileSection title="Skills" count={skills.length}>
      <View style={styles.card}>
        {skills.map((skill, index) => (
          <View key={skill.id} style={[styles.row, index > 0 && styles.divider]}>
            <View style={[styles.tile, { backgroundColor: accentFill(accent, scheme) }]}>
              <Icon icon={BookOpen01Icon} size={18} color={accentInk(accent, scheme)} />
            </View>
            <View style={styles.text}>
              <Text style={styles.title}>{skill.title}</Text>
              {skill.description ? <Text style={styles.description}>{skill.description}</Text> : null}
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
  tile: { width: 38, height: 38, borderRadius: 11, alignItems: "center", justifyContent: "center" },
  text: { flex: 1, minWidth: 0, gap: 2 },
  title: { fontSize: 15.5, lineHeight: 20, fontWeight: "600", color: colors.text },
  description: { fontSize: 14, lineHeight: 19, color: colors.textSecondary },
}));
