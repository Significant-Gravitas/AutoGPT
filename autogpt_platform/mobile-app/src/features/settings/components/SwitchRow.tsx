import { Text, View } from "react-native";

import { Switch } from "@/components/Switch";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  label: string;
  description?: string;
  value: boolean;
  onValueChange: (value: boolean) => void;
}

/** A setting that is on or off, with a line saying what it does. */
export function SwitchRow({ label, description, value, onValueChange }: Props) {
  const styles = useStyles();

  return (
    <View style={styles.row}>
      <View style={styles.text}>
        <Text style={styles.label}>{label}</Text>
        {description ? <Text style={styles.description}>{description}</Text> : null}
      </View>
      <Switch value={value} onValueChange={onValueChange} accessibilityLabel={label} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  text: { flex: 1, gap: 2 },
  label: { ...type.body, color: colors.text },
  description: { ...type.small, color: colors.textSecondary },
}));
