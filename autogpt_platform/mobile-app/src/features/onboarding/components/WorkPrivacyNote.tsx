import { SquareLock02Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export function WorkPrivacyNote() {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.row}>
      <Icon icon={SquareLock02Icon} size={13} color={colors.textTertiary} />
      <Text style={styles.text}>
        Private to you · Saved as Otto memory · Downloadable anytime
      </Text>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
  },
  text: { ...type.small, color: colors.textTertiary },
}));
