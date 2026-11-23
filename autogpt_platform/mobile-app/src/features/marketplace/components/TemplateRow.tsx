import { Add01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint, roleLine } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, radius, useColors } from "@/lib/theme";

const AVATAR_SIZE = 38;
const ACTION_SIZE = 26;

interface Props {
  template: ExpertTemplate;
  isFirst: boolean;
  isOnTeam: boolean;
  isHiring: boolean;
  onOpen: () => void;
  onHire: () => void;
}

export function TemplateRow({ template, isFirst, isOnTeam, isHiring, onOpen, onHire }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const job = roleLine(template) ?? template.name;
  const label = [job, template.name, isOnTeam ? "on your team" : null]
    .filter(Boolean)
    .join(", ");

  // Two siblings, not a button inside a button: a nested one is out of
  // VoiceOver's reach.
  return (
    <View style={[styles.row, !isFirst && styles.divider]}>
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
          background={colors.panelBackground}
          size={AVATAR_SIZE}
        />
        <View style={styles.text}>
          <Text numberOfLines={1} style={styles.job}>
            {job}
          </Text>
          <Text numberOfLines={1} style={styles.who}>
            <Text style={styles.name}>{template.name}</Text>
            {template.tagline?.trim() ? ` · ${template.tagline.trim()}` : null}
          </Text>
        </View>
      </Pressable>
      {isOnTeam ? (
        <View accessibilityLabel="On your team" style={[styles.action, styles.hired]}>
          <Icon icon={Tick02Icon} size={14} color={colors.onHiredBadge} />
        </View>
      ) : (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Hire ${template.name}`}
          accessibilityState={{ busy: isHiring, disabled: isHiring }}
          disabled={isHiring}
          hitSlop={10}
          onPress={onHire}
          style={({ pressed }) => [styles.action, styles.hire, pressed && styles.hirePressed]}
        >
          {isHiring ? (
            <ActivityIndicator size="small" color={colors.text} />
          ) : (
            <Icon icon={Add01Icon} size={16} color={colors.text} />
          )}
        </Pressable>
      )}
    </View>
  );
}

interface CardProps {
  children: ReactNode;
}

/** A rounded card of rows. */
export function RowCard({ children }: CardProps) {
  const styles = useStyles();
  return <View style={styles.card}>{children}</View>;
}

export function PlaceholderRows() {
  const styles = useStyles();
  return (
    <RowCard>
      {[0, 1, 2, 3].map((index) => (
        <View key={index} style={[styles.row, styles.placeholderRow, index > 0 && styles.divider]}>
          <View style={[styles.placeholder, styles.placeholderAvatar]} />
          <View style={styles.text}>
            <View style={[styles.placeholder, styles.placeholderWide]} />
            <View style={[styles.placeholder, styles.placeholderNarrow]} />
          </View>
        </View>
      ))}
    </RowCard>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    marginHorizontal: 14,
    marginBottom: 18,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
    overflow: "hidden",
  },
  row: { flexDirection: "row", alignItems: "center", paddingRight: 12 },
  divider: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.panelDivider },
  open: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 10,
    paddingLeft: 12,
    paddingRight: 10,
  },
  pressed: { opacity: 0.6 },
  text: { flex: 1, minWidth: 0, gap: 1 },
  job: { fontSize: 14, lineHeight: 18, fontWeight: "600", color: colors.text },
  who: { fontSize: 12.5, lineHeight: 16, color: colors.textSecondary },
  name: { fontWeight: "500", color: colors.text },
  action: {
    width: ACTION_SIZE,
    height: ACTION_SIZE,
    borderRadius: ACTION_SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
  },
  // Grey, not black: a column of black buttons outweighs the experts.
  hire: { backgroundColor: colors.panelTabTrack },
  hirePressed: { opacity: 0.7 },
  hired: { backgroundColor: colors.hiredBadge },
  placeholderRow: { gap: 12, paddingVertical: 10, paddingLeft: 12 },
  placeholder: { backgroundColor: colors.panelBackground, borderRadius: radius.pill },
  placeholderAvatar: { width: AVATAR_SIZE, height: AVATAR_SIZE },
  placeholderWide: { height: 11, width: "55%", marginBottom: 5 },
  placeholderNarrow: { height: 9, width: "80%" },
}));
