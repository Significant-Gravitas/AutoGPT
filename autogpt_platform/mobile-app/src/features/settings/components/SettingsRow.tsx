import { ArrowRight01Icon, LinkSquare02Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import { Pressable, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

interface Props {
  label: string;
  icon?: IconSvgElement;
  value?: string;
  /** `page` opens another screen here; `web` leaves the app. */
  trailing?: "page" | "web";
  isDestructive?: boolean;
  onPress?: () => void;
}

export function SettingsRow({
  label,
  icon,
  value,
  trailing,
  isDestructive,
  onPress,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const tint = isDestructive ? colors.error : colors.textSecondary;

  function renderContent() {
    return (
      <>
        {icon ? <Icon icon={icon} size={20} color={tint} /> : null}
        <Text
          numberOfLines={1}
          style={[styles.label, isDestructive && styles.destructive]}
        >
          {label}
        </Text>
        {value ? (
          <Text numberOfLines={1} style={styles.value}>
            {value}
          </Text>
        ) : null}
        {trailing ? (
          <Icon
            icon={trailing === "page" ? ArrowRight01Icon : LinkSquare02Icon}
            size={16}
            color={colors.textTertiary}
          />
        ) : null}
      </>
    );
  }

  if (!onPress) return <View style={styles.row}>{renderContent()}</View>;
  return (
    <Pressable
      accessibilityRole={trailing === "web" ? "link" : "button"}
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      {renderContent()}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  label: { ...type.body, flex: 1, color: colors.text },
  destructive: { color: colors.error },
  value: { ...type.label, fontWeight: "400", color: colors.textSecondary },
}));
