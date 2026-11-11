import { File01Icon } from "@hugeicons/core-free-icons";
import type { UIMessage } from "ai";
import { Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { messageFiles } from "../attachmentHelpers";
import { groupParts } from "../steps";
import { Markdown } from "./Markdown";
import { StepsBlock } from "./StepsBlock";

interface Props {
  message: UIMessage;
  isLive?: boolean;
}

export function MessageBubble({ message, isLive = false }: Props) {
  const styles = useStyles();
  const colors = useColors();
  if (message.role === "user") {
    const text = message.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    const { fileNames } = messageFiles(message.metadata);
    return (
      <View style={styles.userBubble}>
        {text ? <Text style={styles.userText}>{text}</Text> : null}
        {fileNames.length > 0 ? (
          <View style={styles.files}>
            {fileNames.map((name, index) => (
              <View key={`${index}-${name}`} style={styles.file}>
                <Icon icon={File01Icon} size={14} color={colors.textSecondary} />
                <Text style={styles.fileName} numberOfLines={1} ellipsizeMode="middle">
                  {name}
                </Text>
              </View>
            ))}
          </View>
        ) : null}
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
  files: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs, marginTop: spacing.xs },
  file: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    maxWidth: "100%",
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
    borderRadius: radius.sm,
    backgroundColor: colors.surfacePressed,
  },
  fileName: { ...type.small, color: colors.textSecondary, flexShrink: 1 },
  assistant: { gap: spacing.sm },
  thinking: { ...type.small, color: colors.textTertiary },
}));
