import { Add01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_ROLE, OTTO_TINT } from "@/lib/otto";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { expertRoleLabel } from "../expertRoleLabel";

const AVATAR_SIZE = 34;

interface Props {
  placement: { top: number; left: number; width: number; maxHeight: number };
  experts: Expert[];
  /** null is Otto. */
  selectedID: string | null;
  onSelect: (expertID: string | null) => void;
  onCreate: () => void;
  onClose: () => void;
}

export function RecipientMenu({
  placement,
  experts,
  selectedID,
  onSelect,
  onCreate,
  onClose,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
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
        style={[StyleSheet.absoluteFill, styles.backdrop]}
      />
      <ScrollView
        accessibilityRole="menu"
        style={[styles.card, placement]}
        contentContainerStyle={styles.list}
      >
        <Row
          lead={
            <ExpertAvatar
              name={OTTO_NAME}
              source={OTTO_AVATAR_URL}
              tint={OTTO_TINT}
              size={AVATAR_SIZE}
              inset
            />
          }
          name={OTTO_NAME}
          role={OTTO_ROLE}
          isSelected={selectedID === null}
          onPress={() => onSelect(null)}
        />
        {experts.length > 0 ? <View style={styles.separator} /> : null}
        {experts.map((expert) => (
          <Row
            key={expert.id}
            lead={
              <ExpertAvatar
                name={expert.name}
                source={avatarSource(expert)}
                tint={avatarTint(expert)}
                size={AVATAR_SIZE}
              />
            }
            name={expert.name}
            role={
              expert.role?.trim() ? expertRoleLabel(expert.role.trim()) : null
            }
            isSelected={selectedID === expert.id}
            onPress={() => onSelect(expert.id)}
          />
        ))}
        <View style={styles.separator} />
        <Row
          lead={
            <Icon icon={Add01Icon} size={20} color={colors.textSecondary} />
          }
          name="Create an expert"
          isMuted
          onPress={onCreate}
        />
      </ScrollView>
    </Modal>
  );
}

interface RowProps {
  lead: ReactNode;
  name: string;
  role?: string | null;
  isSelected?: boolean;
  isMuted?: boolean;
  onPress: () => void;
}

function Row({ lead, name, role, isSelected, isMuted, onPress }: RowProps) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <Pressable
      accessibilityRole="menuitem"
      accessibilityLabel={role ? `${name}, ${role}` : name}
      accessibilityState={{ selected: !!isSelected }}
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        (isSelected || pressed) && styles.rowSelected,
      ]}
    >
      <View style={styles.lead}>{lead}</View>
      <View style={styles.text}>
        <Text numberOfLines={1} style={[styles.name, isMuted && styles.muted]}>
          {name}
        </Text>
        {role ? (
          <Text numberOfLines={1} style={styles.role}>
            {role}
          </Text>
        ) : null}
      </View>
      {isSelected ? (
        <Icon icon={Tick02Icon} size={18} color={colors.text} />
      ) : null}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  backdrop: { backgroundColor: "rgba(20,20,30,0.08)" },
  card: {
    position: "absolute",
    borderRadius: 22,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.background,
    shadowColor: "#000000",
    shadowOpacity: 0.16,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 12 },
    elevation: 8,
  },
  list: { padding: 6, gap: 2 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: 10,
    paddingVertical: spacing.sm,
    borderRadius: 14,
  },
  rowSelected: { backgroundColor: colors.surfaceRaised },
  lead: { width: AVATAR_SIZE, alignItems: "center" },
  text: { flex: 1, minWidth: 0 },
  name: { fontSize: 16, fontWeight: "500", lineHeight: 21, color: colors.text },
  muted: { fontWeight: "400", color: colors.textSecondary },
  role: { fontSize: 13, lineHeight: 17, color: colors.textTertiary },
  separator: {
    height: StyleSheet.hairlineWidth,
    marginHorizontal: 10,
    marginVertical: spacing.xs,
    backgroundColor: colors.border,
  },
}));
