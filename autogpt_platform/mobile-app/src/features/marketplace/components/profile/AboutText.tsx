import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, Text, type TextLayoutEvent, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, spacing, useColors } from "@/lib/theme";

import { ProfileSection } from "./ProfileSection";

const CLAMPED_LINES = 5;

interface Props {
  text: string;
}

export function AboutText({ text }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const [isExpanded, setIsExpanded] = useState(false);
  const [lineCount, setLineCount] = useState(0);
  const isClampable = lineCount > CLAMPED_LINES;

  function toggle() {
    setIsExpanded(!isExpanded);
  }

  // A hidden full copy reports how many lines the text really takes at this
  // width and text size, so newlines and large type count too.
  function measure(event: TextLayoutEvent) {
    setLineCount(event.nativeEvent.lines.length);
  }

  return (
    <ProfileSection title="About">
      <View>
        <Text
          aria-hidden
          importantForAccessibility="no-hide-descendants"
          onTextLayout={measure}
          style={[styles.text, styles.measure]}
        >
          {text}
        </Text>
        <Text
          numberOfLines={isClampable && !isExpanded ? CLAMPED_LINES : undefined}
          style={styles.text}
        >
          {text}
        </Text>
      </View>
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
  measure: { position: "absolute", top: 0, left: 0, right: 0, opacity: 0 },
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
