import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  message: string;
  onRetry: () => void;
  centered?: boolean;
}

export function RetryNotice({ message, onRetry, centered }: Props) {
  const styles = useStyles();
  return (
    <View style={[styles.notice, centered && styles.centered]}>
      <Text style={styles.message}>{message}</Text>
      <Button label="Try again" onPress={onRetry} variant="secondary" size="small" />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  notice: { alignItems: "flex-start", gap: spacing.sm, paddingHorizontal: spacing.md },
  centered: { alignItems: "center" },
  message: { ...type.small, color: colors.textSecondary },
}));
