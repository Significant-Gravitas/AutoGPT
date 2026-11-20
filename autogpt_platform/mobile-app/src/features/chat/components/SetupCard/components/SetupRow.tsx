import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { ProviderTile } from "@/features/connections/components/ProviderTile";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

interface Props {
  name: string;
  logoID: string | null;
  hint: string | null;
  error: string | null;
  isConnected: boolean;
  actionLabel: string | null;
  /** This row's sign-in is open. */
  isBusy: boolean;
  /** The saved accounts are still loading, so the row can't tell yet. */
  isLoading: boolean;
  /** Another row's sign-in is open. */
  isLocked: boolean;
  onPress: () => void;
}

export function SetupRow({
  name,
  logoID,
  hint,
  error,
  isConnected,
  actionLabel,
  isBusy,
  isLoading,
  isLocked,
  onPress,
}: Props) {
  const styles = useStyles();
  const colors = useColors();

  function renderAction() {
    if (isConnected) {
      return (
        <View style={styles.connected}>
          <Icon icon={CheckmarkCircle02Icon} size={16} color={colors.success} />
          <Text style={styles.connectedText}>Connected</Text>
        </View>
      );
    }
    if (isLoading) return <ActivityIndicator size="small" color={colors.textTertiary} />;
    if (!actionLabel) return null;
    return (
      <Button
        label={isBusy ? "Signing in…" : actionLabel}
        accessibilityLabel={`${actionLabel}: ${name}`}
        onPress={onPress}
        variant={actionLabel === "Sign in" ? "primary" : "outline"}
        size="small"
        disabled={isLocked && !isBusy}
        isLoading={isBusy}
      />
    );
  }

  return (
    <View style={styles.row}>
      <ProviderTile name={name} logoID={logoID} />
      <View style={styles.text}>
        <Text style={styles.name} numberOfLines={1}>
          {name}
        </Text>
        {error ? (
          <Text style={styles.error}>{error}</Text>
        ) : hint ? (
          <Text style={styles.hint} numberOfLines={1}>
            {hint}
          </Text>
        ) : null}
      </View>
      {renderAction()}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  text: { flex: 1, minWidth: 0 },
  name: { ...type.label, color: colors.text },
  hint: { ...type.small, color: colors.textTertiary },
  error: { ...type.small, color: colors.error },
  connected: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  connectedText: { ...type.small, fontWeight: "600", color: colors.success },
}));
