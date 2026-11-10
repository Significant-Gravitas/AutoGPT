import { FlatList, RefreshControl, StyleSheet, Text, View } from "react-native";

import { ScreenHeader } from "@/components/ScreenHeader";
import { ExpertCard, PlaceholderCard } from "@/features/team/components/ExpertCard";
import { avatarSource, roleLine, statusLine } from "@/features/team/helpers";
import { useTeamScreen } from "@/features/team/useTeamScreen";
import type { Expert } from "@/lib/api/experts";
import { colors, spacing, type } from "@/lib/theme";

const appIcon = require("../../../assets/images/icon.png");

export default function TeamScreen() {
  const {
    experts,
    isLoading,
    isError,
    isRefreshing,
    refresh,
    openingID,
    openError,
    openExpert,
    openOtto,
  } = useTeamScreen();

  function handleRefresh() {
    void refresh();
  }

  function renderHeader() {
    return (
      <View style={styles.header}>
        <ExpertCard
          name="Otto"
          avatar={appIcon}
          detail="Ask anything, or hand off a task"
          onPress={openOtto}
        />
        {openError ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {openError}
          </Text>
        ) : null}
      </View>
    );
  }

  function renderEmpty() {
    if (isLoading) {
      return (
        <View style={styles.list}>
          <PlaceholderCard />
          <PlaceholderCard />
          <PlaceholderCard />
        </View>
      );
    }
    return (
      <Text style={styles.notice}>
        {isError
          ? "Couldn't load your team. Pull down to try again."
          : "No experts on your team yet. Hire one on the web and they'll show up here."}
      </Text>
    );
  }

  function renderExpert({ item }: { item: Expert }) {
    function handlePress() {
      void openExpert(item);
    }
    return (
      <ExpertCard
        name={item.name}
        avatar={avatarSource(item)}
        detail={roleLine(item)}
        status={statusLine(item)}
        isOpening={openingID === item.id}
        onPress={handlePress}
      />
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader title="Your team" />
      <FlatList
        data={experts}
        keyExtractor={(expert) => expert.id}
        renderItem={renderExpert}
        ListHeaderComponent={renderHeader()}
        ListEmptyComponent={renderEmpty()}
        ItemSeparatorComponent={Separator}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={handleRefresh}
            tintColor={colors.textTertiary}
          />
        }
      />
    </View>
  );
}

function Separator() {
  return <View style={styles.separator} />;
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  content: { paddingTop: spacing.sm, paddingBottom: spacing.xxl },
  header: { gap: spacing.sm, marginBottom: spacing.lg },
  list: { gap: spacing.sm },
  separator: { height: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
  },
  notice: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: "center",
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.lg,
  },
});
