import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { createStyles, spacing, type } from "@/lib/theme";

import { ProviderTile } from "./ProviderTile";

interface Action {
  label: string;
  variant: "primary" | "outline" | "ghost";
  onPress: () => void;
  isBusy?: boolean;
  isDisabled?: boolean;
}

interface Props {
  name: string;
  detail: string | null;
  logoID?: string | null;
  error?: string | null;
  action?: Action | null;
}

/** A service in a list: its tile, name, a line under it, and one button. */
export function ConnectionRow({ name, detail, logoID, error, action }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.row}>
      <ProviderTile name={name} logoID={logoID} />
      <View style={styles.text}>
        <Text style={styles.name} numberOfLines={1}>
          {name}
        </Text>
        {error ? (
          <Text style={styles.error}>{error}</Text>
        ) : detail ? (
          <Text style={styles.detail} numberOfLines={1}>
            {detail}
          </Text>
        ) : null}
      </View>
      {action ? (
        <Button
          label={action.label}
          accessibilityLabel={`${action.label}: ${name}`}
          onPress={action.onPress}
          variant={action.variant}
          size="small"
          isLoading={action.isBusy}
          disabled={action.isDisabled && !action.isBusy}
        />
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 56,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  text: { flex: 1, minWidth: 0 },
  name: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
  error: { ...type.small, color: colors.error },
}));
