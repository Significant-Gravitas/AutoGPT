import type { ReactNode } from "react";
import { type LayoutChangeEvent, Text, View } from "react-native";

import { createStyles } from "@/lib/theme";

import { areaColor } from "../areaColors";
import type { AreaSection as Section } from "../helpers";
import { RowCard } from "./TemplateRow";

interface Props {
  section: Section;
  onLayout: (event: LayoutChangeEvent) => void;
  children: ReactNode;
}

export function AreaSection({ section, onLayout, children }: Props) {
  const styles = useStyles();
  const { area } = section;
  return (
    <View onLayout={onLayout}>
      <View style={styles.heading}>
        <View style={styles.title}>
          {area ? <View style={[styles.dot, { backgroundColor: areaColor(area) }]} /> : null}
          <Text accessibilityRole="header" style={styles.label}>
            {section.label}
          </Text>
        </View>
        <Text style={styles.count}>{section.templates.length}</Text>
      </View>
      <RowCard>{children}</RowCard>
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
  label: { fontSize: 14.5, lineHeight: 19, fontWeight: "600", color: colors.text },
  count: { fontSize: 11, lineHeight: 15, color: colors.textTertiary },
}));
