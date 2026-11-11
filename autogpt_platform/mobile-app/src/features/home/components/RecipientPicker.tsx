import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_TINT } from "@/lib/otto";
import { createStyles, radius, useColors, useScheme } from "@/lib/theme";

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
            size={24}
          />
        ) : (
          <ExpertAvatar
            name={OTTO_NAME}
            source={OTTO_AVATAR_URL}
            tint={OTTO_TINT}
            size={24}
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
    gap: 6,
    paddingLeft: 5,
    paddingRight: 10,
    paddingVertical: 4,
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
  name: { fontSize: 20, lineHeight: 26, color: colors.textSecondary },
  flipped: { transform: [{ rotate: "180deg" }] },
}));
