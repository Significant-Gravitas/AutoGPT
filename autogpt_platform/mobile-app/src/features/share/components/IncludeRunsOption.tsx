import { Text, View } from "react-native";

import { Switch } from "@/components/Switch";
import { createStyles, radius, spacing, type } from "@/lib/theme";

interface Props {
  value: boolean;
  onChange: (value: boolean) => void;
  disabled: boolean;
}

/** The web's linked-runs option: on, every run from this chat joins the link. */
export function IncludeRunsOption({ value, onChange, disabled }: Props) {
  const styles = useStyles();

  return (
    <View style={styles.row}>
      <View style={styles.text}>
        <Text style={styles.label}>Share workflow runs in this chat</Text>
        <Text style={styles.detail}>
          Includes every run from this conversation, including ones that happen
          after you share.
        </Text>
      </View>
      <Switch
        value={value}
        onValueChange={onChange}
        disabled={disabled}
        accessibilityLabel="Share workflow runs from this chat"
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.background,
  },
  text: { flex: 1, gap: 2 },
  label: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
}));
