import { Text, View } from "react-native";

import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  /** A small heading over the lines, such as "Erased". */
  label?: string;
  lines: string[];
}

export function BulletList({ label, lines }: Props) {
  const styles = useStyles();

  return (
    <View style={styles.list}>
      {label ? <Text style={styles.label}>{label}</Text> : null}
      {lines.map((line) => (
        <View key={line} style={styles.line}>
          <View style={styles.dot} />
          <Text style={styles.text}>{line}</Text>
        </View>
      ))}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  list: { gap: spacing.xs + 2 },
  label: { ...type.small, fontWeight: "500", color: colors.textTertiary },
  line: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm + 2 },
  dot: {
    width: 6,
    height: 6,
    marginTop: 9,
    borderRadius: 3,
    backgroundColor: colors.borderStrong,
  },
  text: { ...type.body, flex: 1, color: colors.textSecondary },
}));
