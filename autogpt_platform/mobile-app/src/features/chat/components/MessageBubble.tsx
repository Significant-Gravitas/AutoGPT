import { isToolUIPart, type UIMessage } from "ai";
import { StyleSheet, Text, View } from "react-native";

import { colors, radius, spacing, type } from "@/lib/theme";

import { Markdown } from "./Markdown";
import { ToolRow } from "./ToolRow";

interface Props {
  message: UIMessage;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    const text = message.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    return (
      <View style={styles.userBubble}>
        <Text style={styles.userText}>{text}</Text>
      </View>
    );
  }

  return (
    <View style={styles.assistant}>
      {message.parts.map((part, index) => {
        if (part.type === "text") {
          return <Markdown key={index} text={part.text} />;
        }
        if (isToolUIPart(part)) return <ToolRow key={index} part={part} />;
        if (part.type === "reasoning" && part.state === "streaming") {
          return (
            <Text key={index} style={styles.thinking}>
              Thinking…
            </Text>
          );
        }
        return null;
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  userBubble: {
    alignSelf: "flex-end",
    maxWidth: "85%",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.lg,
    backgroundColor: colors.surfaceRaised,
  },
  userText: { ...type.body, color: colors.text },
  assistant: { gap: spacing.sm },
  thinking: { ...type.small, color: colors.textTertiary },
});
