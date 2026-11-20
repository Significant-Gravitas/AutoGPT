import { Image } from "expo-image";
import { StyleSheet, Text, TextInput, View } from "react-native";

import { Button } from "@/components/Button";
import { OTTO_AVATAR_URL } from "@/lib/otto";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import {
  OTHER_ROLE,
  OTHER_ROLE_MAX_LENGTH,
  ROLES,
  canContinueRole,
} from "../helpers";
import { OnboardingShell } from "./OnboardingShell";
import { RoleCard } from "./RoleCard";

interface Props {
  role: string;
  otherRole: string;
  onPick: (id: string) => void;
  onChangeOther: (text: string) => void;
  onNext: () => void;
}

// Two cards a row, so a thumb reaches every one.
const ROWS = Array.from({ length: Math.ceil(ROLES.length / 2) }, (_, index) =>
  ROLES.slice(index * 2, index * 2 + 2),
);

export function RoleStep({ role, otherRole, onPick, onChangeOther, onNext }: Props) {
  const styles = useStyles();
  const colors = useColors();

  return (
    <OnboardingShell
      filled={1}
      footer={
        <Button
          label="Next"
          onPress={onNext}
          disabled={!canContinueRole(role, otherRole)}
          fullWidth
        />
      }
    >
      <Image
        source={OTTO_AVATAR_URL}
        style={styles.otto}
        contentFit="contain"
        accessibilityLabel="Otto"
      />
      <Text style={styles.eyebrow}>Welcome to AutoGPT</Text>
      <Text accessibilityRole="header" style={styles.heading}>
        What best describes you?
      </Text>
      <View accessibilityRole="radiogroup" style={styles.grid}>
        {ROWS.map((row) => (
          <View key={row[0].id} style={styles.row}>
            {row.map((item) => (
              <RoleCard
                key={item.id}
                label={item.label}
                icon={item.icon}
                selected={role === item.id}
                onPress={() => onPick(item.id)}
              />
            ))}
          </View>
        ))}
      </View>
      {role === OTHER_ROLE ? (
        <TextInput
          accessibilityLabel="Your role"
          placeholder="Describe your role..."
          placeholderTextColor={colors.textTertiary}
          selectionColor={colors.brand}
          value={otherRole}
          onChangeText={onChangeOther}
          maxLength={OTHER_ROLE_MAX_LENGTH}
          autoFocus
          returnKeyType="done"
          style={styles.field}
        />
      ) : null}
    </OnboardingShell>
  );
}

const useStyles = createStyles((colors) => ({
  otto: { width: 84, height: 84 },
  eyebrow: {
    ...type.small,
    fontWeight: "500",
    color: colors.accentText,
    backgroundColor: colors.brandSoft,
    borderRadius: radius.pill,
    overflow: "hidden",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginTop: spacing.xs,
    marginBottom: spacing.sm,
  },
  heading: { ...type.headline, color: colors.text, textAlign: "center" },
  grid: { alignSelf: "stretch", gap: spacing.md, marginTop: spacing.lg },
  row: { flexDirection: "row", gap: spacing.md },
  field: {
    ...type.body,
    alignSelf: "stretch",
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    color: colors.text,
  },
}));
