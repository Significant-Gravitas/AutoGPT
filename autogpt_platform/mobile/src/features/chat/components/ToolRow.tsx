import { router } from "expo-router";
import { memo } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { Icon } from "../../../design/components/Icon";
import {
  colors,
  MIN_TOUCH,
  radius,
  spacing,
  type,
} from "../../../design/theme";
import { rememberToolPart } from "../tools/detailsStore";
import { presentTool, type ToolPart } from "../tools/presentation";

/**
 * One tool step as a compact, tappable row. It opens read-only details; no
 * card action is offered until that card is built natively.
 */
export const ToolRow = memo(function ToolRow({ tool }: { tool: ToolPart }) {
  const presentation = presentTool(tool);
  const failed = presentation.statusLabel === "Failed";

  function openDetails() {
    const key = rememberToolPart(tool);
    router.push({ pathname: "/tool-details", params: { key } });
  }

  return (
    <Pressable
      onPress={openDetails}
      accessibilityRole="button"
      accessibilityLabel={`${presentation.title}, ${presentation.statusLabel}`}
      accessibilityHint="Shows this step's details"
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      {tool.state === "running" ? (
        <ActivityIndicator size="small" />
      ) : (
        <Icon
          name={presentation.symbol}
          size={16}
          color={failed ? colors.danger : colors.secondaryLabel}
        />
      )}
      <View style={styles.text}>
        <Text style={[type.subheadline, styles.title]} numberOfLines={1}>
          {presentation.title}
        </Text>
        {presentation.summary && tool.state !== "running" ? (
          <Text style={[type.footnote, styles.summary]} numberOfLines={2}>
            {presentation.summary}
          </Text>
        ) : null}
      </View>
      <Text
        style={[
          type.caption,
          { color: failed ? colors.danger : colors.tertiaryLabel },
        ]}
      >
        {presentation.statusLabel}
      </Text>
      <Icon
        name="chevron.right"
        size={12}
        color={colors.tertiaryLabel}
        weight="semibold"
      />
    </Pressable>
  );
});

const styles = StyleSheet.create({
  row: {
    minHeight: MIN_TOUCH,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
    backgroundColor: colors.assistantBubble,
  },
  pressed: { opacity: 0.7 },
  text: { flex: 1, gap: 2 },
  title: { color: colors.label },
  summary: { color: colors.secondaryLabel },
});
