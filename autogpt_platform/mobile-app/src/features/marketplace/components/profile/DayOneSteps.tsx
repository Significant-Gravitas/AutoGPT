import { Text, View } from "react-native";

import type { DayOneItem } from "@/lib/api/experts";
import { createStyles, radius, spacing, useScheme } from "@/lib/theme";

import { accentFill, accentInk } from "../../areaColors";
import { ProfileSection } from "./ProfileSection";

const MARKER_SIZE = 30;

interface Props {
  name: string;
  items: DayOneItem[];
  /** The expert's palette hex, for the numbers. */
  accent: string;
}

/** Day one as a timeline: numbered steps joined by a line, each with its timing. */
export function DayOneSteps({ name, items, accent }: Props) {
  const styles = useStyles();
  const scheme = useScheme();
  if (items.length === 0) return null;

  return (
    <ProfileSection title={`What ${name} sets up on day one`}>
      <View style={styles.card}>
        {items.map((item, index) => (
          <View key={`${index}-${item.title}`} style={[styles.step, index > 0 && styles.later]}>
            {index < items.length - 1 ? <View style={styles.line} /> : null}
            <View style={[styles.marker, { backgroundColor: accentFill(accent, scheme) }]}>
              <Text style={[styles.number, { color: accentInk(accent, scheme) }]}>{index + 1}</Text>
            </View>
            <View style={styles.text}>
              <View style={styles.titleRow}>
                <Text style={styles.title}>{item.title}</Text>
                {item.timing ? <Text style={styles.timing}>{item.timing}</Text> : null}
              </View>
              {item.description ? <Text style={styles.description}>{item.description}</Text> : null}
            </View>
          </View>
        ))}
      </View>
    </ProfileSection>
  );
}

const useStyles = createStyles((colors) => ({
  card: { borderRadius: radius.lg, backgroundColor: colors.panelCard, padding: spacing.lg },
  step: { flexDirection: "row", alignItems: "flex-start", gap: 14 },
  later: { marginTop: 22 },
  // Runs from under this step's number to the next one's.
  line: {
    position: "absolute",
    left: MARKER_SIZE / 2,
    top: MARKER_SIZE + 4,
    bottom: -18,
    width: 1,
    backgroundColor: colors.border,
  },
  marker: {
    width: MARKER_SIZE,
    height: MARKER_SIZE,
    borderRadius: MARKER_SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
  },
  number: { fontSize: 14, lineHeight: 18, fontWeight: "700", fontVariant: ["tabular-nums"] },
  text: { flex: 1, minWidth: 0, paddingTop: 4, gap: 3 },
  titleRow: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm },
  title: { flex: 1, fontSize: 16, lineHeight: 21, fontWeight: "600", color: colors.text },
  timing: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: "600",
    color: colors.textSecondary,
    paddingHorizontal: 9,
    paddingVertical: 2,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.borderBase,
    overflow: "hidden",
  },
  description: { fontSize: 14.5, lineHeight: 20, color: colors.textSecondary },
}));
