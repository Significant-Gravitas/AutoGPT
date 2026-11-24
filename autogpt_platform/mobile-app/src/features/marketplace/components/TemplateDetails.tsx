import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint, roleLine } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { AREAS } from "../helpers";

const AVATAR_SIZE = 96;

const PERKS = [
  "No extra cost. Their work draws on your balance.",
  "Joins your team straight away",
  "Asks before acting on your behalf",
];

interface Props {
  template: ExpertTemplate;
}

export function TemplateDetails({ template }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const role = roleLine(template);
  const areas = AREAS.filter((area) => template.categories.includes(area.key));

  return (
    <View style={styles.details}>
      <View style={styles.top}>
        <ExpertAvatar
          name={template.name}
          source={avatarSource(template)}
          tint={avatarTint(template)}
          size={AVATAR_SIZE}
        />
        <Text style={styles.name}>{template.name}</Text>
        {role ? <Text style={styles.role}>{role}</Text> : null}
        {areas.length ? (
          <View style={styles.tags}>
            {areas.map((area) => (
              <Text key={area.key} style={styles.tag}>
                {area.label}
              </Text>
            ))}
          </View>
        ) : null}
      </View>
      {template.tagline ? <Text style={styles.tagline}>{template.tagline}</Text> : null}
      {template.bio ? <Text style={styles.bio}>{template.bio}</Text> : null}
      {template.bundled_skills.length ? (
        <View style={styles.section}>
          <Text style={styles.heading}>Skills</Text>
          <View style={styles.skills}>
            {template.bundled_skills.map((skill) => (
              <Text key={skill.id} style={styles.skill}>
                {skill.title}
              </Text>
            ))}
          </View>
        </View>
      ) : null}
      <View style={styles.section}>
        <Text style={styles.heading}>What you get</Text>
        {PERKS.map((perk) => (
          <View key={perk} style={styles.perk}>
            <Icon icon={CheckmarkCircle02Icon} size={16} color={colors.success} />
            <Text style={styles.perkText}>{perk}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  details: { gap: spacing.lg },
  top: { alignItems: "center", gap: spacing.xs, paddingTop: spacing.sm },
  name: { ...type.title, color: colors.text, marginTop: spacing.sm },
  role: { ...type.label, fontWeight: "400", color: colors.textSecondary },
  tags: { flexDirection: "row", gap: spacing.xs, marginTop: spacing.xs },
  tag: {
    ...type.small,
    color: colors.textSecondary,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: "hidden",
  },
  tagline: { ...type.body, color: colors.text },
  bio: { ...type.body, color: colors.textSecondary },
  section: { gap: spacing.sm },
  heading: {
    ...type.small,
    fontWeight: "500",
    color: colors.textTertiary,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  skills: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  skill: {
    ...type.small,
    color: colors.text,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.xs,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    overflow: "hidden",
  },
  perk: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm },
  perkText: { ...type.small, flex: 1, color: colors.text },
}));
