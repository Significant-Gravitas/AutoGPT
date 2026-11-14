import { createContext, use, type Ref } from "react";
import type { ScrollViewProps } from "react-native";
import {
  KeyboardChatScrollView,
  type KeyboardChatScrollViewProps,
  type KeyboardChatScrollViewRef,
} from "react-native-keyboard-controller";

type ChatScrollOptions = Pick<
  KeyboardChatScrollViewProps,
  "blankSpace" | "extraContentPadding" | "offset" | "onEndVisible"
>;

export const ChatScrollContext = createContext<ChatScrollOptions>({});

interface Props extends Omit<ScrollViewProps, "ref"> {
  ref?: Ref<KeyboardChatScrollViewRef>;
}

/**
 * The list's scroll view. It moves the thread with the keyboard frame by
 * frame, lifting it only when the end is in view (as ChatGPT does), and
 * leaves room under the floating composer. FlashList needs a stable
 * component here, so the options come through context.
 */
export function ChatScrollView({ ref, ...props }: Props) {
  const options = use(ChatScrollContext);
  return (
    <KeyboardChatScrollView
      ref={ref}
      {...props}
      {...options}
      keyboardLiftBehavior="whenAtEnd"
      keyboardDismissMode="interactive"
      keyboardShouldPersistTaps="handled"
    />
  );
}
