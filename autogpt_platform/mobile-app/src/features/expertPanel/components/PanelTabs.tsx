import { Pressable, Text, View } from "react-native";

import { createStyles, radius, shadows, spacing, useScheme } from "@/lib/theme";

export type PanelTab = "activity" | "library" | "connections";

const TABS: { id: PanelTab; label: string }[] = [
  { id: "activity", label: "Activity" },
  { id: "library", label: "Library" },
  { id: "connections", label: "Connections" },
];

interface Props {
  selected: PanelTab;
  onSelect: (tab: PanelTab) => void;
}

export function PanelTabs({ selected, onSelect }: Props) {
  const styles = useStyles();
  const isLight = useScheme() === "light";

  return (
    <View accessibilityRole="tablist" style={styles.track}>
      {TABS.map((tab) => {
        const isSelected = tab.id === selected;
        return (
          <Pressable
            key={tab.id}
            accessibilityRole="tab"
            accessibilityState={{ selected: isSelected }}
            onPress={() => onSelect(tab.id)}
            style={[
              styles.segment,
              isSelected && styles.selected,
              isSelected && isLight && styles.selectedShadow,
            ]}
          >
            <Text style={[styles.label, isSelected && styles.selectedLabel]}>
              {tab.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  track: {
    flexDirection: "row",
    alignSelf: "center",
    padding: spacing.xs,
    borderRadius: radius.pill,
    backgroundColor: colors.panelTabTrack,
  },
  segment: {
    paddingHorizontal: spacing.lg,
    paddingVertical: 6,
    borderRadius: radius.pill,
  },
  selected: { backgroundColor: colors.panelCard },
  selectedShadow: shadows.selectedTab,
  label: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.textSecondary },
  selectedLabel: { color: colors.text },
}));
