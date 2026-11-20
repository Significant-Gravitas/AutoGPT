import { StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { avatarSource, avatarTint, roleLine } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, radius } from "@/lib/theme";

import { areaColor } from "../areaColors";
import { areaLabel, primaryArea } from "../helpers";

const AVATAR_SIZE = 76;
const FIRST_WEEK_STEPS = 3;

interface Props {
  template: ExpertTemplate;
}

export function TemplateProfile({ template }: Props) {
  const styles = useStyles();
  const area = primaryArea(template);
  const job = roleLine(template) ?? template.name;
  const skills = template.bundled_skills;
  // No first-week plan in the catalogue yet: their first skills stand in.
  // Skills aren't steps in an order, so they go unnumbered.
  const firstWeek = skills.slice(0, FIRST_WEEK_STEPS);

  return (
    <View>
      <View style={styles.top}>
        <ExpertAvatar
          name={template.name}
          source={avatarSource(template)}
          tint={avatarTint(template)}
          size={AVATAR_SIZE}
        />
        {area ? (
          <View style={styles.area}>
            <View style={[styles.dot, { backgroundColor: areaColor(area) }]} />
            <Text style={styles.areaLabel}>{areaLabel(area)}</Text>
          </View>
        ) : null}
        <Text accessibilityRole="header" style={[styles.job, !area && styles.jobAlone]}>
          {job}
        </Text>
        {job !== template.name ? <Text style={styles.name}>{template.name}</Text> : null}
        {template.tagline ? <Text style={styles.tagline}>{template.tagline}</Text> : null}
      </View>
      {firstWeek.length ? (
        <>
          <Heading title="In the first week" />
          <View style={styles.card}>
            {firstWeek.map((skill, index) => (
              <View key={skill.id} style={[styles.step, index > 0 && styles.divider]}>
                <View style={styles.marker}>
                  <View style={[styles.markerDot, area && { backgroundColor: areaColor(area) }]} />
                </View>
                <View style={styles.stepText}>
                  <Text style={styles.stepTitle}>{skill.title}</Text>
                  {skill.description ? (
                    <Text style={styles.stepDetail}>{skill.description}</Text>
                  ) : null}
                </View>
              </View>
            ))}
          </View>
        </>
      ) : null}
      {template.bio ? (
        <>
          <Heading title="About" />
          <View style={[styles.card, styles.bioCard]}>
            <Text style={styles.bio}>{template.bio}</Text>
          </View>
        </>
      ) : null}
      {skills.length ? (
        <>
          <Heading title="Skills" note={String(skills.length)} />
          <View style={[styles.card, styles.chips]}>
            {skills.map((skill) => (
              <Text key={skill.id} style={styles.chip}>
                {skill.title}
              </Text>
            ))}
          </View>
        </>
      ) : null}
    </View>
  );
}

interface HeadingProps {
  title: string;
  note?: string;
}

function Heading({ title, note }: HeadingProps) {
  const styles = useStyles();
  return (
    <View style={styles.heading}>
      <Text accessibilityRole="header" style={styles.headingTitle}>
        {title}
      </Text>
      {note ? <Text style={styles.headingNote}>{note}</Text> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  top: { alignItems: "center", paddingTop: 6, paddingHorizontal: 22, paddingBottom: 18 },
  area: { flexDirection: "row", alignItems: "center", gap: 7, marginTop: 10 },
  dot: { width: 7, height: 7, borderRadius: 3.5 },
  areaLabel: { fontSize: 11.5, lineHeight: 15, color: colors.textSecondary },
  job: {
    fontSize: 18,
    lineHeight: 23,
    fontWeight: "700",
    letterSpacing: -0.2,
    color: colors.text,
    textAlign: "center",
    marginTop: 2,
  },
  jobAlone: { marginTop: 10 },
  name: { fontSize: 12.5, lineHeight: 17, color: colors.textSecondary, textAlign: "center" },
  tagline: {
    fontSize: 12.5,
    lineHeight: 17.5,
    color: colors.text,
    textAlign: "center",
    marginTop: 8,
  },
  heading: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    paddingHorizontal: 18,
    paddingBottom: 6,
  },
  headingTitle: { fontSize: 13.5, lineHeight: 18, fontWeight: "600", color: colors.text },
  headingNote: { fontSize: 11, lineHeight: 15, color: colors.textTertiary },
  card: {
    marginHorizontal: 14,
    marginBottom: 18,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
    overflow: "hidden",
  },
  step: { flexDirection: "row", alignItems: "flex-start", gap: 11, paddingVertical: 11, paddingHorizontal: 13 },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  marker: {
    width: 20,
    height: 20,
    borderRadius: 10,
    marginTop: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.panelBackground,
  },
  markerDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.textTertiary },
  stepText: { flex: 1, gap: 1 },
  stepTitle: { fontSize: 12.5, lineHeight: 17, fontWeight: "600", color: colors.text },
  stepDetail: { fontSize: 11, lineHeight: 15, color: colors.textSecondary },
  bioCard: { padding: 13 },
  bio: { fontSize: 12.5, lineHeight: 17.5, color: colors.text },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 6, padding: 12 },
  chip: {
    fontSize: 11.5,
    lineHeight: 15,
    color: colors.text,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
    backgroundColor: colors.panelBackground,
    overflow: "hidden",
  },
}));
