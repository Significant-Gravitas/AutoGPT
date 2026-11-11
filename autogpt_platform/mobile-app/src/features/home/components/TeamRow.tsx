import { Pressable, ScrollView, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_TINT } from "@/lib/otto";
import { createStyles, spacing } from "@/lib/theme";

interface Props {
  experts: Expert[];
  /** null is Otto. */
  selectedID: string | null;
  onSelect: (expertID: string | null) => void;
}

export function TeamRow({ experts, selectedID, onSelect }: Props) {
  const styles = useStyles();
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}
    >
      <Tile
        name={OTTO_NAME}
        source={OTTO_AVATAR_URL}
        tint={OTTO_TINT}
        inset
        isSelected={selectedID === null}
        onPress={() => onSelect(null)}
      />
      {experts.map((expert) => (
        <Tile
          key={expert.id}
          name={expert.name}
          source={avatarSource(expert)}
          tint={avatarTint(expert)}
          isSelected={selectedID === expert.id}
          onPress={() => onSelect(expert.id)}
        />
      ))}
    </ScrollView>
  );
}

interface TileProps {
  name: string;
  source: string | null;
  tint: string;
  inset?: boolean;
  isSelected: boolean;
  onPress: () => void;
}

function Tile({ name, source, tint, inset, isSelected, onPress }: TileProps) {
  const styles = useStyles();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={name}
      accessibilityState={{ selected: isSelected }}
      onPress={onPress}
      style={styles.tile}
    >
      <View style={[styles.ring, isSelected && styles.ringSelected]}>
        <ExpertAvatar name={name} source={source} size={48} tint={tint} inset={inset} />
      </View>
      <Text
        numberOfLines={1}
        style={[styles.name, isSelected && styles.nameSelected]}
      >
        {name}
      </Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: { gap: spacing.md, paddingHorizontal: spacing.lg },
  tile: { width: 60, alignItems: "center", gap: spacing.xs },
  ring: {
    padding: 2,
    borderRadius: 28,
    borderWidth: 2,
    borderColor: "transparent",
  },
  ringSelected: { borderColor: colors.brand },
  name: { fontSize: 12, lineHeight: 16, color: colors.textSecondary },
  nameSelected: { color: colors.text, fontWeight: "500" },
}));
