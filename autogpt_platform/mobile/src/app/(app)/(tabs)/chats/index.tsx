import { router, Stack, useIsFocused } from "expo-router";
import { useEffect } from "react";
import {
  ActivityIndicator,
  FlatList,
  RefreshControl,
  StyleSheet,
  View,
} from "react-native";

import { PrimaryButton } from "../../../../design/components/Buttons";
import {
  EmptyState,
  ErrorState,
  InlineNotice,
  LoadingState,
} from "../../../../design/components/StateViews";
import { colors, spacing } from "../../../../design/theme";
import {
  useChatList,
  useExperts,
} from "../../../../features/app/AppDataProvider";
import { useSignedIn } from "../../../../features/auth/SessionProvider";
import { ChatRow } from "../../../../features/chats/components/ChatRow";
import {
  conversationHref,
  newConversationHref,
  participantFor,
} from "../../../../features/company/participants";

export default function ChatsScreen() {
  const { server } = useSignedIn();
  const { state, store } = useChatList();
  const { state: experts } = useExperts();

  const focused = useIsFocused();
  useEffect(() => {
    if (focused) void store.refresh();
  }, [focused, store]);

  function openConversation(sessionID: string) {
    router.push(conversationHref(sessionID));
  }

  const toolbar = (
    <>
      <Stack.Toolbar placement="left">
        <Stack.Toolbar.Button
          icon="person.crop.circle"
          accessibilityLabel="Account"
          onPress={() => router.push("/account")}
        />
      </Stack.Toolbar>
      <Stack.Toolbar placement="right">
        <Stack.Toolbar.Menu
          icon="square.and.pencil"
          accessibilityLabel="New conversation"
          title="New conversation with"
        >
          <Stack.Toolbar.MenuAction
            icon="sparkles"
            onPress={() => router.push(newConversationHref(null))}
          >
            Otto
          </Stack.Toolbar.MenuAction>
          {experts.roster.map((expert) => (
            <Stack.Toolbar.MenuAction
              key={expert.id}
              icon="person.crop.circle"
              subtitle={expert.jobTitle ?? expert.role ?? undefined}
              onPress={() => router.push(newConversationHref(expert.id))}
            >
              {expert.name}
            </Stack.Toolbar.MenuAction>
          ))}
        </Stack.Toolbar.Menu>
      </Stack.Toolbar>
    </>
  );

  if (state.status === "idle" || state.status === "loading") {
    return (
      <>
        {toolbar}
        <LoadingState label="Loading conversations" />
      </>
    );
  }
  if (state.status === "error") {
    return (
      <>
        {toolbar}
        <ErrorState
          message={state.error ?? "Couldn't load conversations."}
          onRetry={() => void store.refresh()}
        />
      </>
    );
  }

  return (
    <>
      {toolbar}
      <FlatList
        data={state.sessions}
        keyExtractor={(session) => session.id}
        contentInsetAdjustmentBehavior="automatic"
        style={styles.list}
        renderItem={({ item }) => (
          <ChatRow
            session={item}
            participant={participantFor(item.expertID, experts, server.authUrl)}
            onPress={openConversation}
          />
        )}
        ItemSeparatorComponent={Separator}
        refreshControl={
          <RefreshControl
            refreshing={state.refreshing}
            onRefresh={() => void store.refresh()}
          />
        }
        onEndReached={() => void store.loadMore()}
        onEndReachedThreshold={0.5}
        ListHeaderComponent={
          state.error ? <InlineNotice text={state.error} tone="danger" /> : null
        }
        ListFooterComponent={
          state.loadingMore ? <ActivityIndicator style={styles.footer} /> : null
        }
        ListEmptyComponent={
          <EmptyState
            symbol="bubble.left.and.bubble.right"
            title="No conversations yet"
            message="Ask Otto for anything, or message one of your experts."
            action={
              <PrimaryButton
                title="Message Otto"
                onPress={() => router.push(newConversationHref(null))}
                style={styles.emptyButton}
                testID="message-otto"
              />
            }
          />
        }
        contentContainerStyle={
          state.sessions.length === 0 ? styles.emptyContent : undefined
        }
      />
    </>
  );
}

function Separator() {
  return <View style={styles.separator} />;
}

const styles = StyleSheet.create({
  list: { backgroundColor: colors.background },
  separator: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: colors.separator,
    marginLeft: spacing.lg + 48 + spacing.md,
  },
  footer: { paddingVertical: spacing.lg },
  emptyContent: { flexGrow: 1 },
  emptyButton: { alignSelf: "stretch", marginTop: spacing.md },
});
