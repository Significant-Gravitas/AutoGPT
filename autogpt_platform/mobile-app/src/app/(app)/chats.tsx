import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import {
  ActivityIndicator,
  FlatList,
  RefreshControl,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { ChatGroupSection } from "@/features/chats/components/ChatGroupSection";
import { useAllChats } from "@/features/chats/useAllChats";
import { DRAWER_SESSION_LIMIT } from "@/features/drawer/helpers";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function AllChatsScreen() {
  const {
    groups,
    isTruncated,
    isLoading,
    isError,
    isRefetching,
    refresh,
    goBack,
    openChat,
  } = useAllChats();
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();

  function renderEmpty() {
    if (isLoading) {
      return (
        <View style={styles.centre}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      );
    }
    return (
      <Text style={styles.notice}>
        {isError
          ? "Couldn't load your chats. Pull down to try again."
          : "Your chats will show up here."}
      </Text>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title="All chats"
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={goBack} />}
      />
      <FlatList
        data={groups}
        keyExtractor={(group) => group.key}
        renderItem={({ item }) => (
          <ChatGroupSection
            group={item}
            isTruncated={isTruncated}
            onOpenChat={openChat}
          />
        )}
        ListEmptyComponent={renderEmpty}
        ListFooterComponent={
          isTruncated ? (
            <Text style={styles.notice}>
              Showing your {DRAWER_SESSION_LIMIT} most recent chats.
            </Text>
          ) : undefined
        }
        refreshControl={
          <RefreshControl
            refreshing={isRefetching}
            onRefresh={refresh}
            tintColor={colors.textTertiary}
          />
        }
        contentContainerStyle={[
          styles.content,
          { paddingBottom: insets.bottom + spacing.lg },
        ]}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  content: {
    flexGrow: 1,
    gap: spacing.xl,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
  centre: { flex: 1, alignItems: "center", justifyContent: "center" },
  notice: {
    ...type.small,
    color: colors.textSecondary,
    textAlign: "center",
  },
}));
