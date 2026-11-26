import { ArrowLeft01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { Toast } from "@/components/Toast";
import { FactRow } from "@/features/memory/components/FactRow";
import { UnderstandingCard } from "@/features/memory/components/UnderstandingCard";
import { useMemoryScreen } from "@/features/memory/useMemoryScreen";
import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { SettingsRow } from "@/features/settings/components/SettingsRow";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

const UNAVAILABLE = "Memory isn't available right now.";

export default function MemoryScreen() {
  const screen = useMemoryScreen();
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  function renderRetry() {
    return (
      <Button
        label="Try again"
        variant="secondary"
        size="small"
        isLoading={screen.isRetrying}
        onPress={screen.retry}
      />
    );
  }

  function renderUnavailable() {
    return (
      <View style={styles.inline}>
        <Text style={styles.notice}>{UNAVAILABLE}</Text>
        {renderRetry()}
      </View>
    );
  }

  function renderSectionLoading() {
    return <ActivityIndicator style={styles.sectionLoading} color={colors.textTertiary} />;
  }

  function renderUnderstanding() {
    if (screen.understandingRows) {
      return (
        <UnderstandingCard
          rows={screen.understandingRows}
          onForget={screen.confirmForgetUnderstanding}
        />
      );
    }
    return (
      <SettingsGroup title="What Otto knows about your work">
        {screen.isUnderstandingPending ? renderSectionLoading() : renderUnavailable()}
      </SettingsGroup>
    );
  }

  function renderFacts() {
    if (!screen.facts) {
      return screen.isFactsPending ? renderSectionLoading() : renderUnavailable();
    }
    if (!screen.facts.length) {
      return (
        <Text style={styles.empty}>
          Nothing remembered yet. Memories show up here as you chat.
        </Text>
      );
    }
    return screen.facts.map((fact) => (
      <FactRow
        key={fact.uuid}
        fact={fact}
        onLongPress={screen.openFactMenu}
        onForget={screen.forgetFact}
      />
    ));
  }

  function renderBody() {
    if (screen.isLoading) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    if (screen.isLoadError) {
      return (
        <View style={styles.center}>
          <Text style={styles.notice}>{UNAVAILABLE}</Text>
          {renderRetry()}
        </View>
      );
    }
    return (
      <ScrollView
        contentContainerStyle={[
          styles.content,
          { paddingBottom: insets.bottom + spacing.xxl },
        ]}
      >
        {renderUnderstanding()}
        <SettingsGroup
          title="Recent memories"
          note={screen.facts?.length ? "Hold a memory to forget it." : undefined}
        >
          {renderFacts()}
        </SettingsGroup>
        {screen.overview ? (
          <SettingsGroup>
            <SettingsRow
              icon={Delete02Icon}
              label={screen.isErasing ? "Erasing…" : "Erase all memory"}
              isDestructive
              onPress={screen.isErasing ? undefined : screen.confirmErase}
            />
          </SettingsGroup>
        ) : null}
      </ScrollView>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title="Memory"
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={screen.goBack} />}
      />
      {renderBody()}
      {screen.toast ? <Toast toast={screen.toast} /> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.menuBackground },
  content: { gap: spacing.xl, padding: spacing.lg },
  loading: { marginTop: spacing.xxl },
  sectionLoading: { padding: spacing.lg },
  center: { alignItems: "center", gap: spacing.lg, padding: spacing.xl },
  inline: { alignItems: "center", gap: spacing.md, padding: spacing.lg },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  empty: {
    ...type.body,
    color: colors.textSecondary,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
}));
