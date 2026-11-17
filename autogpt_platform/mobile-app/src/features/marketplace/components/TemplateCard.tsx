import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint, roleLine } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

const AVATAR_SIZE = 48;

interface Props {
  template: ExpertTemplate;
  isOnTeam: boolean;
  isHiring: boolean;
  onOpen: () => void;
  onHire: () => void;
}

export function TemplateCard({ template, isOnTeam, isHiring, onOpen, onHire }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const role = roleLine(template);
  const label = [template.name, role, isOnTeam ? "on your team" : null]
    .filter(Boolean)
    .join(", ");

  // Two siblings, not a button inside a button: a nested one is out of
  // VoiceOver's reach.
  return (
    <View style={styles.card}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={label}
        accessibilityHint="Shows what they do"
        onPress={onOpen}
        style={({ pressed }) => [styles.open, pressed && styles.pressed]}
      >
        <ExpertAvatar
          name={template.name}
          source={avatarSource(template)}
          tint={avatarTint(template)}
          size={AVATAR_SIZE}
        />
        <View style={styles.text}>
          <View style={styles.nameRow}>
            <Text numberOfLines={1} style={styles.name}>
              {template.name}
            </Text>
            {isOnTeam ? <Text style={styles.onTeam}>On your team</Text> : null}
          </View>
          {role ? (
            <Text numberOfLines={1} style={styles.role}>
              {role}
            </Text>
          ) : null}
          {template.tagline ? (
            <Text numberOfLines={2} style={styles.tagline}>
              {template.tagline}
            </Text>
          ) : null}
        </View>
      </Pressable>
      <View style={styles.action}>
        {isOnTeam ? (
          <Icon icon={CheckmarkCircle02Icon} size={20} color={colors.success} />
        ) : (
          <Button
            label="Hire"
            size="small"
            onPress={onHire}
            isLoading={isHiring}
            accessibilityLabel={`Hire ${template.name}`}
          />
        )}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    flexDirection: "row",
    alignItems: "center",
    marginHorizontal: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    overflow: "hidden",
  },
  open: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    padding: spacing.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  action: { paddingRight: spacing.md },
  text: { flex: 1, gap: 2 },
  nameRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.heading, flexShrink: 1, color: colors.text },
  onTeam: { ...type.small, fontWeight: "500", color: colors.success },
  role: { ...type.small, color: colors.textSecondary },
  tagline: { ...type.small, color: colors.textTertiary },
}));
