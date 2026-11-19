import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { createStyles, radius, shadows, type, useColors, useScheme } from "@/lib/theme";

interface Props {
  name: string;
  avatar: string | null;
  tint: string | null;
  /** Opens the chat's panel; without it the pill is only a label. */
  onPress?: () => void;
  /** Read after the name, e.g. "open details". */
  actionLabel?: string;
}

/** Who the chat is with, as a pill in the floating header. */
export function ChatIdentity({ name, avatar, tint, onPress, actionLabel }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const isLight = useScheme() === "light";
  const pill = [styles.pill, isLight && styles.shadow];

  function renderContent() {
    return (
      <>
        <ExpertAvatar name={name} source={avatar} tint={tint} size={28} />
        <Text numberOfLines={1} style={styles.name}>
          {name}
        </Text>
        {onPress ? (
          <Icon icon={ArrowDown01Icon} size={14} color={colors.textTertiary} />
        ) : null}
      </>
    );
  }

  if (!onPress) {
    return (
      <View accessibilityRole="header" style={[pill, styles.still]}>
        {renderContent()}
      </View>
    );
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={actionLabel ? `${name}, ${actionLabel}` : name}
      hitSlop={4}
      onPress={onPress}
      style={({ pressed }) => [pill, pressed && styles.pressed]}
    >
      {renderContent()}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  pill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
    maxWidth: "100%",
    minHeight: 38,
    paddingLeft: 5,
    paddingRight: 12,
    borderRadius: radius.pill,
    backgroundColor: colors.floatingButton,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.floatingRing,
  },
  // Without the chevron, the name sits as far from the edge as the avatar.
  still: { paddingRight: 14 },
  shadow: shadows.floating,
  pressed: { opacity: 0.7 },
  name: { ...type.label, flexShrink: 1, color: colors.text },
}));
