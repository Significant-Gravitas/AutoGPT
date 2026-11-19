import { useRef } from "react";
import {
  type LayoutChangeEvent,
  RefreshControl,
  ScrollView,
  type ScrollViewInstance,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { FLOATING_HEADER_HEIGHT, FloatingHeader } from "@/components/FloatingHeader";
import { Toast } from "@/components/Toast";
import { AreaSection } from "@/features/marketplace/components/AreaSection";
import { MarketplaceSearchField } from "@/features/marketplace/components/MarketplaceSearchField";
import { TeamCard } from "@/features/marketplace/components/TeamCard";
import {
  PlaceholderRows,
  RowCard,
  TemplateRow,
} from "@/features/marketplace/components/TemplateRow";
import type { Area } from "@/features/marketplace/helpers";
import { useMarketplaceScreen } from "@/features/marketplace/useMarketplaceScreen";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles, spacing, useColors } from "@/lib/theme";

export default function MarketplaceScreen() {
  const screen = useMarketplaceScreen();
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const scroll = useRef<ScrollViewInstance>(null);
  // Where each area's section starts, so an open seat can scroll to it.
  const sectionTops = useRef<Partial<Record<Area, number>>>({});
  const headerHeight = insets.top + FLOATING_HEADER_HEIGHT;

  function handleRefresh() {
    void screen.refresh();
  }

  function scrollToArea(area: Area) {
    const top = sectionTops.current[area];
    if (top === undefined) return;
    scroll.current?.scrollTo({ y: Math.max(0, top - headerHeight), animated: true });
  }

  function renderRow(template: ExpertTemplate, index: number) {
    return (
      <TemplateRow
        key={template.id}
        template={template}
        isFirst={index === 0}
        isOnTeam={screen.isOnTeam(template)}
        isHiring={screen.isHiring(template.id)}
        onOpen={() => screen.openTemplate(template)}
        onHire={() => void screen.hire(template)}
      />
    );
  }

  function renderNotice(title: string, detail?: string) {
    return (
      <View accessibilityLiveRegion="polite" style={styles.notice}>
        <Text style={styles.noticeTitle}>{title}</Text>
        {detail ? <Text style={styles.noticeDetail}>{detail}</Text> : null}
      </View>
    );
  }

  function renderBody() {
    if (screen.isLoading) return <PlaceholderRows />;
    if (screen.isError) {
      return renderNotice("Couldn't load the marketplace", "Pull down to try again.");
    }
    if (!screen.hasTemplates) return renderNotice("No experts yet", "Pull down to check again.");
    if (screen.isSearching) {
      if (!screen.matches.length) return renderNotice("No experts match.");
      return <RowCard background={colors.panelCard}>{screen.matches.map(renderRow)}</RowCard>;
    }
    return (
      <>
        <TeamCard
          team={screen.team}
          openSeats={screen.openSeats}
          openingID={screen.openingID}
          onOpenExpert={(expert) => void screen.openChat(expert)}
          onOpenSeat={scrollToArea}
        />
        {screen.sections.map((section) => (
          <AreaSection
            key={section.area ?? "other"}
            section={section}
            onLayout={(event: LayoutChangeEvent) => {
              if (section.area) sectionTops.current[section.area] = event.nativeEvent.layout.y;
            }}
          >
            {section.templates.map(renderRow)}
          </AreaSection>
        ))}
      </>
    );
  }

  return (
    <View style={styles.screen}>
      <ScrollView
        ref={scroll}
        contentContainerStyle={[
          styles.content,
          { paddingTop: headerHeight, paddingBottom: insets.bottom + spacing.xxl },
        ]}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="on-drag"
        refreshControl={
          <RefreshControl
            refreshing={screen.isRefreshing}
            onRefresh={handleRefresh}
            tintColor={colors.textTertiary}
            progressViewOffset={headerHeight}
          />
        }
      >
        <Text accessibilityRole="header" style={styles.title}>
          Hire an expert
        </Text>
        <MarketplaceSearchField value={screen.query} onChangeText={screen.setQuery} />
        {renderBody()}
      </ScrollView>
      {/* After the list, so it floats over it. */}
      <FloatingHeader fadeColor={colors.panelBackground} />
      {screen.toast ? <Toast toast={screen.toast} /> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.panelBackground },
  content: { flexGrow: 1 },
  title: {
    fontSize: 23,
    lineHeight: 28,
    fontWeight: "700",
    letterSpacing: -0.4,
    color: colors.text,
    paddingHorizontal: 18,
    paddingTop: 10,
    paddingBottom: 10,
  },
  notice: { alignItems: "center", gap: spacing.xs, paddingHorizontal: spacing.xl, paddingTop: spacing.lg },
  noticeTitle: { fontSize: 15, lineHeight: 20, fontWeight: "500", color: colors.text, textAlign: "center" },
  noticeDetail: { fontSize: 13, lineHeight: 18, color: colors.textSecondary, textAlign: "center" },
}));
