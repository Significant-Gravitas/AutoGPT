import { FlatList, RefreshControl, Text, View } from "react-native";

import { ScreenHeader } from "@/components/ScreenHeader";
import { Toast } from "@/components/Toast";
import { DrawerSearchField } from "@/features/drawer/components/DrawerSearchField";
import { AreaChips } from "@/features/marketplace/components/AreaChips";
import { TemplateCard } from "@/features/marketplace/components/TemplateCard";
import { useMarketplaceScreen } from "@/features/marketplace/useMarketplaceScreen";
import { PlaceholderCard } from "@/features/team/components/ExpertCard";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function MarketplaceScreen() {
  const screen = useMarketplaceScreen();
  const styles = useStyles();
  const colors = useColors();

  function handleRefresh() {
    void screen.refresh();
  }

  function renderEmpty() {
    if (screen.isLoading) {
      return (
        <View style={styles.list}>
          <PlaceholderCard />
          <PlaceholderCard />
          <PlaceholderCard />
          <PlaceholderCard />
        </View>
      );
    }
    const { title, detail } = screen.isError
      ? { title: "Couldn't load the marketplace", detail: "Pull down to try again." }
      : screen.empty;
    return (
      <View accessibilityLiveRegion="polite" style={styles.notice}>
        <Text style={styles.noticeTitle}>{title}</Text>
        <Text style={styles.noticeDetail}>{detail}</Text>
      </View>
    );
  }

  function renderTemplate({ item }: { item: ExpertTemplate }) {
    function handleOpen() {
      screen.openTemplate(item);
    }
    function handleHire() {
      void screen.hire(item);
    }
    return (
      <TemplateCard
        template={item}
        isOnTeam={screen.isOnTeam(item)}
        isHiring={screen.isHiring(item.id)}
        onOpen={handleOpen}
        onHire={handleHire}
      />
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader title="Marketplace" />
      <View style={styles.search}>
        <DrawerSearchField
          value={screen.query}
          onChangeText={screen.setQuery}
          placeholder="Search experts"
        />
      </View>
      <View style={styles.chips}>
        <AreaChips selected={screen.area} onSelect={screen.setArea} />
      </View>
      <FlatList
        data={screen.templates}
        keyExtractor={(template) => template.id}
        renderItem={renderTemplate}
        ListEmptyComponent={renderEmpty()}
        ItemSeparatorComponent={Separator}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="on-drag"
        refreshControl={
          <RefreshControl
            refreshing={screen.isRefreshing}
            onRefresh={handleRefresh}
            tintColor={colors.textTertiary}
          />
        }
      />
      {screen.toast ? <Toast toast={screen.toast} /> : null}
    </View>
  );
}

function Separator() {
  const styles = useStyles();
  return <View style={styles.separator} />;
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  search: { paddingHorizontal: spacing.lg, paddingTop: spacing.xs },
  chips: { paddingVertical: spacing.md },
  content: { paddingBottom: spacing.xxl * 3 },
  list: { gap: spacing.sm },
  separator: { height: spacing.sm },
  notice: { alignItems: "center", gap: spacing.xs, paddingHorizontal: spacing.xl, paddingTop: spacing.xl },
  noticeTitle: { ...type.label, color: colors.text, textAlign: "center" },
  noticeDetail: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
