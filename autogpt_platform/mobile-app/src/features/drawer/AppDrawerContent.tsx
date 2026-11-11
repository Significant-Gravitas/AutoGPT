import {
  Logout01Icon,
  PencilEdit02Icon,
  Settings01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { DrawerContentComponentProps } from "expo-router/drawer";
import {
  ActivityIndicator,
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing, type, useColors } from "@/lib/theme";

import { DrawerSearchField } from "./components/DrawerSearchField";
import { ExpertRecentRow } from "./components/ExpertRecentRow";
import { RecentHeading } from "./components/RecentHeading";
import { SearchResultRow } from "./components/SearchResultRow";
import { DrawerRow } from "./DrawerRow";
import { useAppDrawer } from "./useAppDrawer";

interface Props {
  navigation: DrawerContentComponentProps["navigation"];
}

export function AppDrawerContent({ navigation }: Props) {
  const insets = useSafeAreaInsets();
  const {
    query,
    setQuery,
    isSearching,
    groups,
    matches,
    isLoading,
    isError,
    refetch,
    isRefetching,
    email,
    activeSessionID,
    openChat,
    newChat,
    openTeam,
    openAllChats,
    openSettings,
    handleSignOut,
  } = useAppDrawer(navigation);
  const styles = useStyles();
  const colors = useColors();

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

  return (
    <View
      style={[
        styles.container,
        { paddingTop: insets.top + spacing.sm, paddingBottom: insets.bottom + spacing.sm },
      ]}
    >
      <View style={styles.section}>
        <DrawerSearchField value={query} onChangeText={setQuery} />
        <DrawerRow icon={PencilEdit02Icon} label="New chat" onPress={newChat} />
        <DrawerRow icon={UserGroupIcon} label="Your team" onPress={openTeam} />
      </View>
      {isSearching ? (
        <FlatList
          data={matches}
          keyExtractor={(match) => match.session.id}
          renderItem={({ item }) => (
            <SearchResultRow
              match={item}
              isActive={item.session.id === activeSessionID}
              onOpenChat={openChat}
            />
          )}
          ListEmptyComponent={<Text style={styles.notice}>No chats match.</Text>}
          keyboardShouldPersistTaps="handled"
          style={styles.list}
          contentContainerStyle={styles.searchContent}
        />
      ) : (
        <>
          <RecentHeading onOpenAllChats={openAllChats} />
          <FlatList
            data={groups}
            keyExtractor={(group) => group.key}
            renderItem={({ item }) => (
              <ExpertRecentRow
                group={item}
                isActive={item.sessions.some(
                  (session) => session.id === activeSessionID,
                )}
                onOpenChat={openChat}
              />
            )}
            ListEmptyComponent={renderEmpty}
            refreshControl={
              <RefreshControl
                refreshing={isRefetching}
                onRefresh={handleRefresh}
                tintColor={colors.textTertiary}
              />
            }
            style={styles.list}
            contentContainerStyle={styles.listContent}
          />
        </>
      )}
      <View style={[styles.section, styles.footer]}>
        {email ? (
          <Text numberOfLines={1} style={styles.email}>
            {email}
          </Text>
        ) : null}
        <DrawerRow icon={Settings01Icon} label="Settings" onPress={openSettings} />
        <DrawerRow icon={Logout01Icon} label="Sign out" onPress={handleSignOut} />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  container: {
    flex: 1,
    paddingHorizontal: spacing.sm,
    backgroundColor: colors.background,
  },
  section: { gap: spacing.xs },
  list: { flex: 1 },
  listContent: { flexGrow: 1, gap: 2 },
  searchContent: { flexGrow: 1, gap: 2, paddingTop: spacing.sm },
  notice: {
    ...type.small,
    color: colors.textSecondary,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
  },
  footer: {
    paddingTop: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  email: {
    ...type.small,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.xs,
  },
}));
