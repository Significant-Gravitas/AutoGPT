import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

interface Props {
  onPress: () => void;
}

/** The dark pill floating at the foot of the drawer. */
export function NewChatButton({ onPress }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.pill, pressed && styles.pressed]}
    >
      <Icon icon={PencilEdit02Icon} size={22} color={colors.background} />
      <Text style={styles.label}>New chat</Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  pill: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    height: 48,
    paddingLeft: spacing.lg,
    paddingRight: spacing.xl,
    borderRadius: radius.pill,
    backgroundColor: colors.text,
    shadowColor: "#000000",
    shadowOpacity: 0.18,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  pressed: { opacity: 0.8 },
  label: { fontSize: 17, lineHeight: 22, fontWeight: "600", color: colors.background },
}));
