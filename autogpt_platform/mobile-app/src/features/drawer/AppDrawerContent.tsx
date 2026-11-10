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

import { colors, spacing, type } from "@/lib/theme";

import { DrawerRow } from "./DrawerRow";
import { useAppDrawer } from "./useAppDrawer";

interface Props {
  navigation: DrawerContentComponentProps["navigation"];
}

export function AppDrawerContent({ navigation }: Props) {
  const insets = useSafeAreaInsets();
  const {
    sessions,
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
      <Text style={styles.heading}>Chats</Text>
      <FlatList
        data={sessions}
        keyExtractor={(session) => session.id}
        renderItem={({ item }) => (
          <DrawerRow
            label={item.title ?? "New chat"}
            isActive={item.id === activeSessionID}
            isBusy={item.is_processing}
            onPress={() => openChat(item.id)}
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
  listContent: { flexGrow: 1, gap: spacing.xs },
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
