import { Pressable, Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import type { Scheme } from "../helpers";

interface Props {
  schemes: Scheme[];
  value: string;
  onChange: (value: string) => void;
}

/** The kind of server a login is for, when there is more than one. */
export function SchemePicker({ schemes, value, onChange }: Props) {
  const styles = useStyles();
  return (
    <View accessibilityRole="radiogroup" style={styles.row}>
      {schemes.map((scheme) => {
        const isSelected = scheme.value === value;
        return (
          <Pressable
            key={scheme.value}
            accessibilityRole="radio"
            accessibilityState={{ checked: isSelected }}
            onPress={() => onChange(scheme.value)}
            style={[styles.chip, isSelected && styles.selected]}
          >
            <Text style={[styles.label, isSelected && styles.selectedLabel]}>
              {scheme.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.borderBase,
  },
  selected: { backgroundColor: colors.action, borderColor: colors.action },
  label: { ...type.small, color: colors.text },
  selectedLabel: { color: colors.onAction },
}));
