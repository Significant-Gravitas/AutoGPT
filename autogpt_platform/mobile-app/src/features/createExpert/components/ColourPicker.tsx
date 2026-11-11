import { Pressable, Text, View } from "react-native";

import { createStyles, spacing, type } from "@/lib/theme";

import { colourFamilies, type Family } from "../helpers";

const SWATCH_SIZE = 32;

interface Props {
  selected: Family;
  onSelect: (family: Family) => void;
}

export function ColourPicker({ selected, onSelect }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.section}>
      <Text style={styles.label}>Colour</Text>
      <View style={styles.row}>
        {colourFamilies.map(({ family, label, hex }) => {
          const isSelected = family === selected;
          return (
            <Pressable
              key={family}
              accessibilityRole="button"
              accessibilityLabel={label}
              accessibilityState={{ selected: isSelected }}
              hitSlop={4}
              onPress={() => onSelect(family)}
              style={[styles.ring, isSelected && styles.ringSelected]}
            >
              <View style={[styles.swatch, { backgroundColor: hex }]} />
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
  row: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  ring: {
    padding: 2,
    borderRadius: SWATCH_SIZE,
    borderWidth: 2,
    borderColor: "transparent",
  },
  ringSelected: { borderColor: colors.text },
  swatch: {
    width: SWATCH_SIZE,
    height: SWATCH_SIZE,
    borderRadius: SWATCH_SIZE / 2,
  },
}));
