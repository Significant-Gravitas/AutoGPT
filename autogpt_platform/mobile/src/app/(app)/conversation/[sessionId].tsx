import { Stack, useLocalSearchParams } from "expo-router";
import { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { KeyboardStickyView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import {
  ErrorState,
  LoadingState,
} from "../../../design/components/StateViews";
import { colors } from "../../../design/theme";
import { useExperts } from "../../../features/app/AppDataProvider";
import { useSignedIn } from "../../../features/auth/SessionProvider";
import { Composer } from "../../../features/chat/components/Composer";
import { ConversationIntro } from "../../../features/chat/components/ConversationIntro";
import { MessageList } from "../../../features/chat/components/MessageList";
import { useConversation } from "../../../features/chat/useConversation";
import { participantFor } from "../../../features/company/participants";

const STOPPABLE = new Set(["streaming", "reconnecting", "queued", "offline"]);

export default function ConversationScreen() {
  const params = useLocalSearchParams<{
    sessionId: string;
    expertId?: string;
  }>();
  const { server } = useSignedIn();
  const { state: experts, directory } = useExperts();
  const insets = useSafeAreaInsets();
  const { snapshot, controller } = useConversation(
    params.sessionId && params.sessionId !== "new" ? params.sessionId : null,
    params.expertId ?? null,
  );
  const [actions] = useState(() => ({
    send: (text: string) => void controller.send(text),
    stop: () => void controller.stop(),
    retrySend: (messageID: string) => void controller.retry(messageID),
    discard: (messageID: string) => controller.discard(messageID),
    loadOlder: () => void controller.loadOlder(),
    retry: () => void controller.refresh(),
  }));

  useEffect(() => {
    directory.ensureLoaded();
  }, [directory]);

  const participant = participantFor(
    snapshot.expertID,
    experts,
    server.authUrl,
  );

  let body;
  if (snapshot.loadState === "loading") {
    body = <LoadingState label="Loading conversation" />;
  } else if (snapshot.loadState === "error") {
    body = (
      <ErrorState
        message={snapshot.loadError ?? "Couldn't open this conversation."}
        onRetry={() => controller.reload()}
      />
    );
  } else if (
    snapshot.items.length === 0 &&
    snapshot.turn === "idle" &&
    !snapshot.notice
  ) {
    body = <ConversationIntro participant={participant} />;
  } else {
    body = (
      <MessageList
        snapshot={snapshot}
        participantName={participant.name}
        onRetrySend={actions.retrySend}
        onDiscard={actions.discard}
        onLoadOlder={actions.loadOlder}
        onRetry={actions.retry}
      />
    );
  }

  return (
    <View style={styles.screen}>
      <Stack.Screen options={{ title: participant.name }} />
      {body}
      <KeyboardStickyView offset={{ closed: 0, opened: insets.bottom }}>
        <Composer
          placeholder={`Message ${participant.name}`}
          readOnlyReason={participant.readOnlyReason}
          busy={snapshot.turn !== "idle" || snapshot.loadState !== "ready"}
          canStop={STOPPABLE.has(snapshot.turn)}
          bottomInset={insets.bottom}
          onSend={actions.send}
          onStop={actions.stop}
        />
      </KeyboardStickyView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
});
