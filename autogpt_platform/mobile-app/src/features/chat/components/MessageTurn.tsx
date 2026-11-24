import type { ChatStatus } from "ai";
import { Pressable, View, type LayoutChangeEvent } from "react-native";
import Animated, { FadeIn, FadeInDown } from "react-native-reanimated";

import { createStyles, spacing } from "@/lib/theme";

import { openMessageActions } from "../messageActions";
import { isLocalMessage, messageText } from "../messageText";
import type { Turn } from "../turns";
import { MessageBubble } from "./MessageBubble";
import { ThinkingIndicator } from "./ThinkingIndicator";

interface Props {
  turn: Turn;
  sessionID: string;
  isLast: boolean;
  status: ChatStatus;
  onSend: (text: string) => boolean;
  onLayout?: (event: LayoutChangeEvent) => void;
}

export function MessageTurn({ turn, sessionID, isLast, status, onSend, onLayout }: Props) {
  const styles = useStyles();
  const isBusy = status === "submitted" || status === "streaming";
  const isWaiting = isLast && status === "submitted";

  return (
    <View onLayout={onLayout} style={styles.turn}>
      {turn.items.map(({ message, replyText }, index) => {
        const isFinal = isLast && index === turn.items.length - 1;
        // Only what was sent or streamed just now fades in, never history.
        const isFresh = isLast && isBusy && isLocalMessage(message, sessionID);
        const entering = !isFresh
          ? undefined
          : message.role === "user"
            ? FadeInDown.duration(220)
            : FadeIn.duration(220);

        function handleLongPress() {
          void openMessageActions(messageText(message));
        }

        return (
          <Animated.View key={message.id} entering={entering}>
            <Pressable delayLongPress={350} onLongPress={handleLongPress}>
              <MessageBubble
                message={message}
                isLive={isBusy && isFinal && message.role === "assistant"}
                replyText={replyText}
                isBusy={isBusy}
                onSend={onSend}
              />
            </Pressable>
          </Animated.View>
        );
      })}
      {isWaiting ? <ThinkingIndicator /> : null}
    </View>
  );
}

const useStyles = createStyles(() => ({
  turn: { gap: spacing.lg, paddingTop: spacing.lg },
}));
