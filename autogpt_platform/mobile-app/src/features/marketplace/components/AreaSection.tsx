import type { ReactNode } from "react";
import { type LayoutChangeEvent, Text, View } from "react-native";

import { createStyles, useScheme } from "@/lib/theme";

import { areaColor, areaTint } from "../areaColors";
import type { AreaSection as Section } from "../helpers";
import { RowCard } from "./TemplateRow";

interface Props {
  section: Section;
  onLayout: (event: LayoutChangeEvent) => void;
  children: ReactNode;
}

export function AreaSection({ section, onLayout, children }: Props) {
  const styles = useStyles();
  const scheme = useScheme();
  return (
    <View onLayout={onLayout}>
      <View style={styles.heading}>
        <View style={styles.title}>
          <View style={[styles.dot, { backgroundColor: areaColor(section.area) }]} />
          <Text accessibilityRole="header" style={styles.label}>
            {section.label}
          </Text>
        </View>
        {section.isCovered ? null : <Text style={styles.note}>Your team has no one here</Text>}
      </View>
      <RowCard background={areaTint(section.area, scheme)}>{children}</RowCard>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  heading: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 8,
    paddingHorizontal: 18,
    paddingBottom: 6,
  },
  title: { flexDirection: "row", alignItems: "center", gap: 7 },
  dot: { width: 7, height: 7, borderRadius: 3.5 },
  label: { fontSize: 13.5, lineHeight: 18, fontWeight: "600", color: colors.text },
  note: { fontSize: 11, lineHeight: 15, color: colors.textTertiary },
}));
