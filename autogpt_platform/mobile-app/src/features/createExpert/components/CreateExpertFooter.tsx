import { Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  error: string | null;
  canCreate: boolean;
  isCreating: boolean;
  onCancel: () => void;
  onCreate: () => void;
}

export function CreateExpertFooter({
  error,
  canCreate,
  isCreating,
  onCancel,
  onCreate,
}: Props) {
  const styles = useStyles();
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.footer, { paddingBottom: insets.bottom + spacing.md }]}>
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <View style={styles.buttons}>
        <View style={styles.half}>
          <Button label="Cancel" variant="secondary" fullWidth onPress={onCancel} />
        </View>
        <View style={styles.half}>
          <Button
            label="Create"
            fullWidth
            isLoading={isCreating}
            disabled={!canCreate}
            onPress={onCreate}
          />
        </View>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  footer: {
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    backgroundColor: colors.background,
  },
  error: { ...type.small, color: colors.error },
  buttons: { flexDirection: "row", gap: spacing.md },
  half: { flex: 1 },
}));
