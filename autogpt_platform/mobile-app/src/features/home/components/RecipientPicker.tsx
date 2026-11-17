import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { router } from "expo-router";
import { Pressable, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_TINT } from "@/lib/otto";
import { createStyles, radius, spacing, type, useColors, useScheme } from "@/lib/theme";

import { useRecipientMenu } from "../useRecipientMenu";
import { RecipientMenu } from "./RecipientMenu";

interface Props {
  experts: Expert[];
  /** null is Otto. */
  selected: Expert | null;
  onSelect: (expertID: string | null) => void;
}

export function RecipientPicker({ experts, selected, onSelect }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  const { pillRef, placement, isOpen, open, close } = useRecipientMenu();
  const name = selected?.name ?? OTTO_NAME;

  function handleSelect(expertID: string | null) {
    close();
    onSelect(expertID);
  }

  function handleCreate() {
    close();
    router.navigate("/create-expert");
  }

  return (
    <>
      <Pressable
        ref={pillRef}
        collapsable={false}
        accessibilityRole="button"
        accessibilityLabel={`Talking to ${name}, change`}
        accessibilityState={{ expanded: isOpen }}
        onPress={open}
        style={({ pressed }) => [
          styles.pill,
          scheme === "light" && styles.shadow,
          pressed && styles.pressed,
        ]}
      >
        {selected ? (
          <ExpertAvatar
            name={selected.name}
            source={avatarSource(selected)}
            tint={avatarTint(selected)}
            size={30}
          />
        ) : (
          <ExpertAvatar
            name={OTTO_NAME}
            source={OTTO_AVATAR_URL}
            tint={OTTO_TINT}
            size={30}
            inset
          />
        )}
        <Text numberOfLines={1} style={styles.name}>
          {name}
        </Text>
        <View style={isOpen && styles.flipped}>
          <Icon icon={ArrowDown01Icon} size={16} color={colors.textSecondary} />
        </View>
      </Pressable>
      {placement ? (
        <RecipientMenu
          placement={placement}
          experts={experts}
          selectedID={selected?.id ?? null}
          onSelect={handleSelect}
          onCreate={handleCreate}
          onClose={close}
        />
      ) : null}
    </>
  );
}

const useStyles = createStyles((colors) => ({
  pill: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    height: 40,
    paddingLeft: 4,
    paddingRight: spacing.md,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  shadow: {
    shadowColor: "#000000",
    shadowOpacity: 0.05,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  pressed: { opacity: 0.6 },
  name: { ...type.label, fontSize: 16, color: colors.text },
  flipped: { transform: [{ rotate: "180deg" }] },
}));
