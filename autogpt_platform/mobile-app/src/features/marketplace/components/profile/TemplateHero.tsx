import { Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, spacing, useScheme } from "@/lib/theme";

import { accentFill, accentInk, areaColor } from "../../areaColors";
import { areaLabel, primaryArea } from "../../helpers";

const PORTRAIT_SIZE = 96;

interface Props {
  template: ExpertTemplate;
  stats: string[];
}

/** The portrait on the page's left edge, the name, the job, the area, the tagline and what comes with the hire. */
export function TemplateHero({ template, stats }: Props) {
  const styles = useStyles();
  const scheme = useScheme();
  const area = primaryArea(template);
  const job = template.job_title || template.role;
  const areaHex = area ? areaColor(area) : null;

  return (
    <View style={styles.hero}>
      <View style={styles.portrait}>
        <ExpertAvatar
          name={template.name}
          source={avatarSource(template)}
          tint={avatarTint(template)}
          size={PORTRAIT_SIZE}
        />
      </View>
      <Text accessibilityRole="header" style={styles.name}>
        {template.name}
      </Text>
      {job ? <Text style={styles.job}>{job}</Text> : null}
      {area && areaHex ? (
        <View style={[styles.area, { backgroundColor: accentFill(areaHex, scheme) }]}>
          <View style={[styles.dot, { backgroundColor: areaHex }]} />
          <Text style={[styles.areaLabel, { color: accentInk(areaHex, scheme) }]}>
            {areaLabel(area)}
          </Text>
        </View>
      ) : null}
      {template.tagline ? <Text style={styles.tagline}>{template.tagline}</Text> : null}
      {stats.length ? (
        <View accessibilityLabel={`Comes with ${stats.join(", ")}`} style={styles.stats}>
          {stats.map((stat) => (
            <Text key={stat} style={styles.stat}>
              {stat}
            </Text>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  hero: { paddingHorizontal: 20, paddingTop: spacing.xs, alignItems: "flex-start" },
  // A ring in the page colour, so the portrait stands off the wash.
  portrait: {
    borderRadius: PORTRAIT_SIZE,
    borderWidth: 4,
    borderColor: colors.panelBackground,
    shadowColor: "#101828",
    shadowOpacity: 0.18,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
  name: {
    marginTop: 14,
    fontSize: 28,
    lineHeight: 33,
    fontWeight: "700",
    letterSpacing: -0.5,
    color: colors.text,
  },
  job: { fontSize: 16, lineHeight: 21, fontWeight: "500", color: colors.textSecondary },
  area: {
    marginTop: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    height: 26,
    paddingHorizontal: 10,
    borderRadius: 13,
  },
  dot: { width: 7, height: 7, borderRadius: 3.5 },
  areaLabel: { fontSize: 13, lineHeight: 16, fontWeight: "600" },
  tagline: { marginTop: 12, fontSize: 16, lineHeight: 23, color: colors.text },
  stats: { marginTop: 14, flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  stat: {
    fontSize: 13,
    lineHeight: 17,
    fontWeight: "600",
    color: colors.textSecondary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.panelCard,
    overflow: "hidden",
  },
}));
