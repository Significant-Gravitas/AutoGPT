import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  name: string;
  isOnTeam: boolean;
  isHiring: boolean;
  isOpeningChat: boolean;
  error: string | null;
  onHire: () => void;
  onChat: () => void;
}

export function HireBar({ name, isOnTeam, isHiring, isOpeningChat, error, onHire, onChat }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.bar}>
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      {isOnTeam ? (
        <>
          <View style={styles.onTeam}>
            <Icon icon={CheckmarkCircle02Icon} size={16} color={colors.success} />
            <Text style={styles.onTeamText}>On your team</Text>
          </View>
          <Button
            label={`Chat with ${name}`}
            variant="secondary"
            fullWidth
            isLoading={isOpeningChat}
            onPress={onChat}
          />
        </>
      ) : (
        <Button
          label={isHiring ? `Hiring ${name}…` : `Hire ${name}`}
          fullWidth
          isLoading={isHiring}
          onPress={onHire}
        />
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  bar: { gap: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    backgroundColor: colors.errorSurface,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    overflow: "hidden",
  },
  onTeam: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
  },
  onTeamText: { ...type.label, color: colors.success },
}));
