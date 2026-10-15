import {
  ActivityIndicator,
  FlatList,
  StyleSheet,
  type ScrollViewProps,
} from "react-native";
import { KeyboardChatScrollView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { spacing } from "../../../design/theme";
import type { ConversationSnapshot } from "../conversationController";
import type { TimelineItem } from "../model/timeline";
import { AssistantMessage, WorkUpdate } from "./AssistantMessage";
import { TurnStatus } from "./TurnStatus";
import { UserMessage } from "./UserMessage";

interface Props {
  snapshot: ConversationSnapshot;
  participantName: string;
  onRetrySend: (messageID: string) => void;
  onDiscard: (messageID: string) => void;
  onLoadOlder: () => void;
  onRetry: () => void;
}

/**
 * Newest message at the bottom via an inverted list, so streaming text stays
 * anchored. The scroll view lifts with the keyboard only when the newest
 * message is in view, leaving older reading undisturbed.
 */
export function MessageList({
  snapshot,
  participantName,
  onRetrySend,
  onDiscard,
  onLoadOlder,
  onRetry,
}: Props) {
  const canRetry = snapshot.turn === "idle" && snapshot.loadState === "ready";

  function renderItem({ item }: { item: TimelineItem }) {
    switch (item.kind) {
      case "user":
        return (
          <UserMessage
            item={item}
            canRetry={canRetry}
            onRetry={onRetrySend}
            onDiscard={onDiscard}
          />
        );
      case "assistant":
        return <AssistantMessage item={item} />;
      case "work":
        return <WorkUpdate text={item.text} />;
    }
  }

  return (
    <FlatList
      inverted
      data={[...snapshot.items].reverse()}
      keyExtractor={(item) => item.key}
      renderItem={renderItem}
      renderScrollComponent={renderScrollComponent}
      ListHeaderComponent={
        <TurnStatus
          turn={snapshot.turn}
          statusText={snapshot.statusText}
          notice={snapshot.notice}
          participantName={participantName}
          onRetry={onRetry}
        />
      }
      ListFooterComponent={
        snapshot.loadingOlder ? (
          <ActivityIndicator style={styles.older} />
        ) : null
      }
      onEndReached={snapshot.hasOlder ? onLoadOlder : undefined}
      onEndReachedThreshold={0.4}
      keyboardDismissMode="interactive"
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={styles.content}
      windowSize={15}
      testID="message-list"
    />
  );
}

function renderScrollComponent(props: ScrollViewProps) {
  return <ChatScrollView {...props} />;
}

function ChatScrollView(props: ScrollViewProps) {
  const insets = useSafeAreaInsets();
  return (
    <KeyboardChatScrollView
      {...props}
      inverted
      offset={insets.bottom}
      keyboardLiftBehavior="whenAtEnd"
      automaticallyAdjustContentInsets={false}
      contentInsetAdjustmentBehavior="never"
    />
  );
}

const styles = StyleSheet.create({
  content: { paddingTop: spacing.xs, paddingBottom: spacing.md },
  older: { paddingVertical: spacing.lg },
});
