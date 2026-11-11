import {
  KeyboardAvoidingView,
  Platform,
  RefreshControl,
  ScrollView,
  Text,
  View,
} from "react-native";

import { ScreenHeader } from "@/components/ScreenHeader";
import { Composer } from "@/features/chat/components/Composer";
import { PromptChips } from "@/features/chat/components/PromptChips";
import { Greeting } from "@/features/home/components/Greeting";
import { NeedsYouCard } from "@/features/home/components/NeedsYouCard";
import { NowNextCard } from "@/features/home/components/NowNextCard";
import { TeamRow } from "@/features/home/components/TeamRow";
import { useHomeScreen } from "@/features/home/useHomeScreen";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function HomeScreen() {
  const home = useHomeScreen();
  const styles = useStyles();
  const colors = useColors();

  function handleSend(text: string) {
    void home.submit(text);
  }

  function handleRefresh() {
    void home.refresh();
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
        refreshControl={
          <RefreshControl
            refreshing={home.isRefreshing}
            onRefresh={handleRefresh}
            tintColor={colors.textTertiary}
          />
        }
      >
        <View style={styles.sections}>
          <Greeting heading={home.heading} subline={home.subline} />
          <NeedsYouCard items={home.attention} />
          <NowNextCard active={home.activeTask} upcoming={home.upcomingTask} />
        </View>
        <TeamRow
          experts={home.experts}
          selectedID={home.selectedID}
          onSelect={home.select}
        />
      </ScrollView>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.bottom}
      >
        {home.error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {home.error}
          </Text>
        ) : null}
        <PromptChips prompts={home.prompts} onPick={handleSend} />
        <Composer
          onSend={home.submit}
          isDisabled={home.isStarting}
          placeholder={home.placeholder}
        />
      </KeyboardAvoidingView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  scroll: { flex: 1 },
  // The team row sits at the foot of the scroll area, above the chips.
  content: {
    flexGrow: 1,
    justifyContent: "space-between",
    gap: spacing.xl,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
  },
  sections: { gap: spacing.md, paddingHorizontal: spacing.lg },
  bottom: { gap: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
  },
}));
