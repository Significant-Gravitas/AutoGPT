import { Pressable, Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import { areaOptions, type Family } from "../helpers";

const DOT_SIZE = 12;

interface Props {
  selected: Family;
  onSelect: (family: Family) => void;
}

export function AreaPicker({ selected, onSelect }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.section}>
      <Text style={styles.label}>Area</Text>
      <View style={styles.row}>
        {areaOptions.map(({ family, label, hex }) => {
          const isSelected = family === selected;
          return (
            <Pressable
              key={family}
              accessibilityRole="button"
              accessibilityState={{ selected: isSelected }}
              onPress={() => onSelect(family)}
              style={[styles.chip, isSelected && styles.chipSelected]}
            >
              <View style={[styles.dot, { backgroundColor: hex }]} />
              <Text style={[styles.name, isSelected && styles.nameSelected]}>
                {label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.sm },
  label: { ...type.small, color: colors.textSecondary },
  row: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    height: 36,
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  chipSelected: {
    borderColor: colors.text,
    backgroundColor: colors.surfaceRaised,
  },
  dot: { width: DOT_SIZE, height: DOT_SIZE, borderRadius: DOT_SIZE / 2 },
  name: { fontSize: 14, lineHeight: 20, fontWeight: "500", color: colors.textSecondary },
  nameSelected: { color: colors.text },
}));
