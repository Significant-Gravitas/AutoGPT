import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { colors, radius, spacing, type } from "@/lib/theme";

const AVATAR_SIZE = 48;

interface Props {
  name: string;
  avatar: string | number | null;
  detail?: string | null;
  status?: string | null;
  isOpening?: boolean;
  onPress: () => void;
}

export function ExpertCard({
  name,
  avatar,
  detail,
  status,
  isOpening,
  onPress,
}: Props) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={name}
      accessibilityState={{ busy: isOpening }}
      onPress={onPress}
      style={({ pressed }) => [styles.card, pressed && styles.pressed]}
    >
      <ExpertAvatar name={name} source={avatar} size={AVATAR_SIZE} />
      <View style={styles.text}>
        <Text numberOfLines={1} style={styles.name}>
          {name}
        </Text>
        {detail ? (
          <Text numberOfLines={1} style={styles.detail}>
            {detail}
          </Text>
        ) : null}
        {status ? (
          <Text numberOfLines={1} style={styles.status}>
            {status}
          </Text>
        ) : null}
      </View>
      {isOpening ? <ActivityIndicator color={colors.textTertiary} /> : null}
    </Pressable>
  );
}

export function PlaceholderCard() {
  return (
    <View style={styles.card}>
      <View style={[styles.avatar, styles.placeholder]} />
      <View style={styles.text}>
        <View style={[styles.bar, styles.placeholder, styles.barWide]} />
        <View style={[styles.bar, styles.placeholder]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    marginHorizontal: spacing.lg,
    padding: spacing.md,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  avatar: {
    width: AVATAR_SIZE,
    height: AVATAR_SIZE,
    borderRadius: AVATAR_SIZE / 2,
  },
  text: { flex: 1, gap: 2 },
  name: { ...type.heading, color: colors.text },
  detail: { ...type.small, color: colors.textSecondary },
  status: { ...type.small, color: colors.textTertiary },
  placeholder: { backgroundColor: colors.surfaceRaised },
  bar: { height: 12, width: "40%", borderRadius: radius.sm },
  barWide: { width: "60%", height: 16 },
});
