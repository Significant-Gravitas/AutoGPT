import { UserIcon } from "@hugeicons/core-free-icons";
import { Pressable, ScrollView, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { firstName } from "@/features/marketplace/helpers";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { createStyles, radius, spacing, useColors } from "@/lib/theme";

import type { OwnerChip } from "../../libraryHelpers";

interface Props {
  chip: OwnerChip;
  team: Expert[];
  onChange: (chip: OwnerChip) => void;
}

/** One row of chips: everything, what you uploaded, or one expert's work. */
export function OwnerChips({ chip, team, onChange }: Props) {
  const styles = useStyles();
  const colors = useColors();

  function renderChip(key: string, label: string, isOn: boolean, next: OwnerChip, lead?: React.ReactNode) {
    return (
      <Pressable
        key={key}
        accessibilityRole="button"
        accessibilityState={{ selected: isOn }}
        onPress={() => onChange(next)}
        style={[styles.chip, !lead && styles.plain, isOn && styles.on]}
      >
        {lead}
        <Text style={[styles.label, isOn && styles.labelOn]}>{label}</Text>
      </Pressable>
    );
  }

  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
      {renderChip("all", "All", chip.kind === "all", { kind: "all" })}
      {renderChip(
        "you",
        "You",
        chip.kind === "you",
        { kind: "you" },
        <View style={styles.you}>
          <Icon icon={UserIcon} size={13} color={colors.textSecondary} />
        </View>,
      )}
      {team.map((expert) =>
        renderChip(
          expert.id,
          firstName(expert.name),
          chip.kind === "expert" && chip.id === expert.id,
          { kind: "expert", id: expert.id },
          <ExpertAvatar
            name={expert.name}
            source={avatarSource(expert)}
            tint={avatarTint(expert)}
            background={colors.panelBackground}
            size={24}
          />,
        ),
      )}
    </ScrollView>
  );
}

const useStyles = createStyles((colors) => ({
  row: { gap: spacing.sm, paddingHorizontal: 14, paddingBottom: spacing.lg },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    height: 32,
    paddingLeft: 4,
    paddingRight: 12,
    borderRadius: radius.pill,
    backgroundColor: colors.panelCard,
  },
  plain: { paddingLeft: 14, paddingRight: 14 },
  on: { backgroundColor: colors.actionStrong },
  label: { fontSize: 13, lineHeight: 17, fontWeight: "500", color: colors.text },
  labelOn: { color: colors.onActionStrong },
  you: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.panelTabTrack,
  },
}));
