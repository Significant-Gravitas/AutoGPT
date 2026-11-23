import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, Text } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { ProfileSection } from "./ProfileSection";

const CLAMPED_LINES = 5;
// Roughly what fits in five lines on a phone; shorter text shows whole.
const CLAMPED_LENGTH = 240;

interface Props {
  text: string;
}

export function AboutText({ text }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_LENGTH;

  function toggle() {
    setIsExpanded(!isExpanded);
  }

  return (
    <ProfileSection title="About">
      <Text
        numberOfLines={isClampable && !isExpanded ? CLAMPED_LINES : undefined}
        style={styles.text}
      >
        {text}
      </Text>
      {isClampable ? (
        <Pressable accessibilityRole="button" onPress={toggle} hitSlop={8} style={styles.more}>
          <Text style={styles.moreLabel}>{isExpanded ? "Show less" : "Read more"}</Text>
          <Icon
            icon={isExpanded ? ArrowUp01Icon : ArrowDown01Icon}
            size={15}
            color={colors.textSecondary}
          />
        </Pressable>
      ) : null}
    </ProfileSection>
  );
}

const useStyles = createStyles((colors) => ({
  text: { paddingHorizontal: spacing.xs, fontSize: 16, lineHeight: 24, color: colors.text },
  more: {
    marginTop: spacing.sm,
    marginLeft: spacing.xs,
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 4,
  },
  moreLabel: { fontSize: 14.5, lineHeight: 19, fontWeight: "600", color: colors.textSecondary },
}));
