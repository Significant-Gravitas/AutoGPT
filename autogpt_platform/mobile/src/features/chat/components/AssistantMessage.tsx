import { memo, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "../../../design/components/Icon";
import {
  colors,
  MIN_TOUCH,
  radius,
  spacing,
  type,
} from "../../../design/theme";
import type { AssistantPart, TimelineItem } from "../model/timeline";
import { MarkdownText } from "./MarkdownText";
import { ToolRow } from "./ToolRow";

type AssistantItem = Extract<TimelineItem, { kind: "assistant" }>;

/** Replies read as full-width text; steps and notices sit inline where they happened. */
export const AssistantMessage = memo(function AssistantMessage({
  item,
}: {
  item: AssistantItem;
}) {
  return (
    <View style={styles.container}>
      {item.parts.map((part) => (
        <PartView key={part.key} part={part} />
      ))}
    </View>
  );
});

function PartView({ part }: { part: AssistantPart }) {
  switch (part.kind) {
    case "text":
      return <MarkdownText text={part.text} />;
    case "reasoning":
      return <Reasoning text={part.text} streaming={part.streaming} />;
    case "tool":
      return <ToolRow tool={part.tool} />;
    case "error":
      return (
        <View style={styles.error} accessibilityRole="alert">
          <Icon
            name="exclamationmark.triangle.fill"
            size={16}
            color={colors.danger}
          />
          <Text selectable style={[type.subheadline, styles.errorText]}>
            {part.text}
            {part.retryable ? " Sending your message again may work." : ""}
          </Text>
        </View>
      );
    case "system":
      return <Text style={[type.footnote, styles.system]}>{part.text}</Text>;
  }
}

function Reasoning({ text, streaming }: { text: string; streaming: boolean }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <View>
      <Pressable
        onPress={() => setExpanded((open) => !open)}
        accessibilityRole="button"
        accessibilityState={{ expanded }}
        accessibilityLabel={streaming ? "Thinking" : "Thought process"}
        style={styles.reasoningHeader}
      >
        <Icon name="brain" size={14} color={colors.secondaryLabel} />
        <Text style={[type.footnote, styles.reasoningLabel]}>
          {streaming ? "Thinking…" : "Thought process"}
        </Text>
        <Icon
          name={expanded ? "chevron.up" : "chevron.down"}
          size={11}
          color={colors.tertiaryLabel}
        />
      </Pressable>
      {expanded ? <MarkdownText text={text} tone="secondary" /> : null}
    </View>
  );
}

/** A post an expert made about its own work, outside an ordinary reply. */
export const WorkUpdate = memo(function WorkUpdate({ text }: { text: string }) {
  return (
    <View style={styles.workContainer}>
      <View style={styles.workCard}>
        <View style={styles.workHeader}>
          <Icon name="checklist" size={14} color={colors.tint} />
          <Text style={[type.footnote, styles.workLabel]}>Work update</Text>
        </View>
        <MarkdownText text={text} />
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  error: {
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.md,
    borderRadius: radius.md,
    backgroundColor: colors.assistantBubble,
  },
  errorText: { flex: 1, color: colors.label },
  system: { color: colors.secondaryLabel, textAlign: "center" },
  reasoningHeader: {
    minHeight: MIN_TOUCH,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    alignSelf: "flex-start",
  },
  reasoningLabel: { color: colors.secondaryLabel },
  workContainer: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  workCard: {
    borderRadius: radius.md,
    backgroundColor: colors.assistantBubble,
    padding: spacing.md,
    gap: spacing.sm,
  },
  workHeader: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  workLabel: { color: colors.secondaryLabel, fontWeight: "600" },
});
