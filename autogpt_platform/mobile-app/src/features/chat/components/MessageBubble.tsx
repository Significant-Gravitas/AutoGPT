import { File01Icon } from "@hugeicons/core-free-icons";
import type { UIMessage } from "ai";
import { Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { ToastState } from "@/components/useToast";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { messageFiles } from "../attachmentHelpers";
import { messageNotice } from "../notice";
import { groupParts } from "../steps";
import { Markdown } from "./Markdown";
import { NoticeCard } from "./NoticeCard/NoticeCard";
import { OnboardingCard } from "./OnboardingCard/OnboardingCard";
import { SetupCard } from "./SetupCard/SetupCard";
import { StepsBlock } from "./StepsBlock";

interface Props {
  message: UIMessage;
  sessionID: string;
  /** Set in an expert's chat. */
  expertID: string | null;
  isLive?: boolean;
  /** Text of the next message, or null when this one is the last. */
  replyText: string | null;
  isBusy: boolean;
  onSend: (text: string) => boolean;
  onToast: (toast: ToastState) => void;
}

export function MessageBubble({
  message,
  sessionID,
  expertID,
  isLive = false,
  replyText,
  isBusy,
  onSend,
  onToast,
}: Props) {
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

  // A notice has no parts of its own; a stream resumed onto it still shows.
  const notice = messageNotice(message.metadata);
  const blocks = groupParts(message.parts);
  return (
    <View style={styles.assistant}>
      {notice ? (
        <NoticeCard notice={notice} sessionID={sessionID} expertID={expertID} onToast={onToast} />
      ) : null}
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
        if (block.kind === "onboarding") {
          return (
            <OnboardingCard
              key={block.key}
              part={block.part}
              replyText={replyText}
              isBusy={isBusy}
              onSend={onSend}
            />
          );
        }
        if (block.kind === "setup") {
          return (
            <SetupCard
              key={block.key}
              sessionID={sessionID}
              toolCallID={block.key}
              setup={block.setup}
              replyText={replyText}
              isBusy={isBusy}
              onSend={onSend}
            />
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
