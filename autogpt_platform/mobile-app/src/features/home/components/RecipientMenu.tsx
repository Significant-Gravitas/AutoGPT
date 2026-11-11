import { Modal, Pressable, ScrollView, StyleSheet, Text } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_TINT } from "@/lib/otto";
import { createStyles, spacing } from "@/lib/theme";

interface Props {
  placement: { top: number; left: number; maxHeight: number };
  experts: Expert[];
  /** null is Otto. */
  selectedID: string | null;
  onSelect: (expertID: string | null) => void;
  onClose: () => void;
}

export function RecipientMenu({
  placement,
  experts,
  selectedID,
  onSelect,
  onClose,
}: Props) {
  const styles = useStyles();
  return (
    <Modal
      transparent
      visible
      animationType="fade"
      statusBarTranslucent
      navigationBarTranslucent
      onRequestClose={onClose}
    >
      <Pressable
        accessibilityLabel="Close menu"
        onPress={onClose}
        style={StyleSheet.absoluteFill}
      />
      <ScrollView
        accessibilityRole="menu"
        style={[styles.card, placement]}
        contentContainerStyle={styles.list}
      >
        <Row
          name={OTTO_NAME}
          source={OTTO_AVATAR_URL}
          tint={OTTO_TINT}
          inset
          isSelected={selectedID === null}
          onPress={() => onSelect(null)}
        />
        {experts.map((expert) => (
          <Row
            key={expert.id}
            name={expert.name}
            source={avatarSource(expert)}
            tint={avatarTint(expert)}
            isSelected={selectedID === expert.id}
            onPress={() => onSelect(expert.id)}
          />
        ))}
      </ScrollView>
    </Modal>
  );
}

interface RowProps {
  name: string;
  source: string | null;
  tint: string;
  inset?: boolean;
  isSelected: boolean;
  onPress: () => void;
}

function Row({ name, source, tint, inset, isSelected, onPress }: RowProps) {
  const styles = useStyles();
  return (
    <Pressable
      accessibilityRole="menuitem"
      accessibilityLabel={name}
      accessibilityState={{ selected: isSelected }}
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        (isSelected || pressed) && styles.rowSelected,
      ]}
    >
      <ExpertAvatar name={name} source={source} size={28} tint={tint} inset={inset} />
      <Text numberOfLines={1} style={styles.name}>
        {name}
      </Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    position: "absolute",
    minWidth: 200,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
    shadowColor: "#000000",
    shadowOpacity: 0.12,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  list: { padding: spacing.xs, gap: 2 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 10,
    paddingVertical: spacing.sm,
    borderRadius: 10,
  },
  rowSelected: { backgroundColor: colors.surfaceRaised },
  name: { flexShrink: 1, fontSize: 16, lineHeight: 22, color: colors.text },
}));
