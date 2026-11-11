import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

interface Props {
  onOpenAllChats: () => void;
}

export function RecentHeading({ onOpenAllChats }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.row}>
      <Text style={styles.label}>Recent</Text>
      <Pressable
        accessibilityRole="link"
        hitSlop={{ top: 12, bottom: 12, left: 16, right: 12 }}
        onPress={onOpenAllChats}
        style={({ pressed }) => [styles.link, pressed && styles.pressed]}
      >
        <Text style={styles.label}>All chats</Text>
        <Icon icon={ArrowRight01Icon} size={12} color={colors.textTertiary} />
      </Pressable>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.md,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
  },
  label: { ...type.small, color: colors.textTertiary },
  link: { flexDirection: "row", alignItems: "center", gap: 2 },
  pressed: { opacity: 0.6 },
}));
