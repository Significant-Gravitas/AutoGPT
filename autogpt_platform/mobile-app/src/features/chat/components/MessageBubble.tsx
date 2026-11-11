import type { UIMessage } from "ai";
import { Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

import { groupParts } from "../steps";
import { Markdown } from "./Markdown";
import { StepsBlock } from "./StepsBlock";

interface Props {
  message: UIMessage;
  isLive?: boolean;
}

export function MessageBubble({ message, isLive = false }: Props) {
  const styles = useStyles();
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

  const blocks = groupParts(message.parts);
  return (
    <View style={styles.assistant}>
      {blocks.map((block, index) => {
        if (block.kind === "text") {
          return <Markdown key={block.key} text={block.text} />;
        }
        if (block.kind === "thinking") {
          return (
            <Text key={block.key} style={styles.thinking}>
              Thinking…
            </Text>
          );
        }
        return (
          <StepsBlock
            key={block.key}
            steps={block.steps}
            isLive={isLive && index === blocks.length - 1}
          />
        );
      })}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
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
}));
