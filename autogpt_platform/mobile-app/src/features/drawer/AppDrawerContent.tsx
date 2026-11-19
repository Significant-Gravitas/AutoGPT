import {
  BubbleChatIcon,
  Store01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { DrawerContentComponentProps } from "expo-router/drawer";
import { ActivityIndicator, FlatList, RefreshControl, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { DrawerHeader } from "./components/DrawerHeader";
import { ExpertRecentRow } from "./components/ExpertRecentRow";
import { NewChatButton } from "./components/NewChatButton";
import { RecentHeading } from "./components/RecentHeading";
import { SearchResultRow } from "./components/SearchResultRow";
import { UserMenu } from "./components/UserMenu";
import { DrawerRow, ROW_INSET } from "./DrawerRow";
import { useAppDrawer } from "./useAppDrawer";

// The floating buttons' height, plus room to scroll the last row clear of them.
const FOOTER_CLEARANCE = 48 + spacing.xl;

interface Props {
  navigation: DrawerContentComponentProps["navigation"];
}

export function AppDrawerContent({ navigation }: Props) {
  const insets = useSafeAreaInsets();
  const {
    query,
    setQuery,
    isSearchOpen,
    openSearch,
    closeSearch,
    isSearching,
    groups,
    matches,
    isLoading,
    isError,
    refetch,
    isRefetching,
    email,
    userName,
    userImage,
    activeSessionID,
    openChat,
    newChat,
    openTeam,
    openMarketplace,
    openAllChats,
    openSettings,
    handleSignOut,
  } = useAppDrawer(navigation);
  const styles = useStyles();
  const colors = useColors();
  const footerBottom = insets.bottom + spacing.sm;
  const listPadding = { paddingBottom: footerBottom + FOOTER_CLEARANCE };

  function handleRefresh() {
    void refetch();
  }

  function renderEmpty() {
    if (isLoading) return <ActivityIndicator color={colors.textTertiary} />;
    return (
      <Text style={styles.notice}>
        {isError
          ? "Couldn't load your chats. Pull down to try again."
          : "Your chats will show up here."}
      </Text>
    );
  }

  function renderListHeader() {
    if (isSearchOpen) return <RecentHeading />;
    return (
      <>
        <View style={styles.nav}>
          <DrawerRow icon={UserGroupIcon} label="Your team" onPress={openTeam} />
          <DrawerRow icon={Store01Icon} label="Marketplace" onPress={openMarketplace} />
          <DrawerRow icon={BubbleChatIcon} label="All chats" onPress={openAllChats} />
        </View>
        <RecentHeading />
      </>
    );
  }

  function renderRefresh() {
    return (
      <RefreshControl
        refreshing={isRefetching}
        onRefresh={handleRefresh}
        tintColor={colors.textTertiary}
      />
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top + spacing.sm }]}>
      <DrawerHeader
        isSearchOpen={isSearchOpen}
        query={query}
        onChangeQuery={setQuery}
        onOpenSearch={openSearch}
        onCloseSearch={closeSearch}
      />
      {isSearching ? (
        <FlatList
          // A key per mode, so each list starts at its top.
          key="search"
          data={matches}
          keyExtractor={(match) => match.session.id}
          renderItem={({ item }) => (
            <SearchResultRow
              match={item}
              isActive={item.session.id === activeSessionID}
              onOpenChat={openChat}
            />
          )}
          ListEmptyComponent={
            isLoading || isError ? (
              renderEmpty
            ) : (
              <Text style={styles.notice}>No chats match.</Text>
            )
          }
          refreshControl={renderRefresh()}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="on-drag"
          style={styles.list}
          contentContainerStyle={[styles.searchContent, listPadding]}
        />
      ) : (
        <FlatList
          key="recent"
          data={groups}
          keyExtractor={(group) => group.key}
          renderItem={({ item }) => (
            <ExpertRecentRow
              group={item}
              isActive={item.sessions.some((session) => session.id === activeSessionID)}
              onOpenChat={openChat}
            />
          )}
          ListHeaderComponent={renderListHeader()}
          ListEmptyComponent={renderEmpty}
          refreshControl={renderRefresh()}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="on-drag"
          style={styles.list}
          contentContainerStyle={[styles.listContent, listPadding]}
        />
      )}
      {/* Floats over the list, which scrolls underneath. */}
      <View style={[styles.footer, { bottom: footerBottom }]}>
        <NewChatButton onPress={newChat} />
        <UserMenu
          name={userName ?? null}
          email={email ?? null}
          image={userImage ?? null}
          onOpenSettings={openSettings}
          onSignOut={handleSignOut}
        />
      </View>
    </View>
  );
}

// No background: the drawer is see-through, so the page's shadow falls on the
// menu's grey behind it (the app layout's root).
const useStyles = createStyles((colors) => ({
  container: { flex: 1 },
  nav: { paddingTop: spacing.sm },
  list: { flex: 1 },
  listContent: { flexGrow: 1, gap: 2, paddingHorizontal: spacing.md },
  searchContent: {
    flexGrow: 1,
    gap: 2,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
  },
  notice: {
    ...type.small,
    fontSize: 14,
    color: colors.textSecondary,
    paddingHorizontal: ROW_INSET,
    paddingTop: spacing.sm,
  },
  footer: {
    position: "absolute",
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.xl,
    pointerEvents: "box-none",
  },
}));
