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

import { DRAWER_SESSION_LIMIT } from "@/features/drawer/helpers";
import { colors, spacing, type } from "@/lib/theme";

import { ExpertChatGroup } from "./components/ExpertChatGroup";
import { DrawerRow } from "./DrawerRow";
import { useAppDrawer } from "./useAppDrawer";

interface Props {
  navigation: DrawerContentComponentProps["navigation"];
}

export function AppDrawerContent({ navigation }: Props) {
  const insets = useSafeAreaInsets();
  const {
    groups,
    isTruncated,
    isLoading,
    isError,
    refetch,
    isRefetching,
    email,
    activeSessionID,
    openChat,
    newChat,
    openTeam,
    openSettings,
    handleSignOut,
  } = useAppDrawer(navigation);

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
        <DrawerRow icon={PencilEdit02Icon} label="New chat" onPress={newChat} />
        <DrawerRow icon={UserGroupIcon} label="Your team" onPress={openTeam} />
      </View>
      <Text style={styles.heading}>Recent chats</Text>
      <FlatList
        data={groups}
        keyExtractor={(group) => group.key}
        renderItem={({ item }) => (
          <ExpertChatGroup
            group={item}
            isTruncated={isTruncated}
            activeSessionID={activeSessionID}
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
            onRefresh={handleRefresh}
            tintColor={colors.textTertiary}
          />
        }
        style={styles.list}
        contentContainerStyle={styles.listContent}
      />
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

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: spacing.sm,
    backgroundColor: colors.background,
  },
  section: { gap: spacing.xs },
  heading: {
    ...type.small,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
  },
  list: { flex: 1 },
  listContent: { flexGrow: 1, gap: spacing.sm },
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
});
