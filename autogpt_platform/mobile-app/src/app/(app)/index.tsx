import {
  RefreshControl,
  ScrollView,
  Text,
  View,
} from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { ScreenHeader } from "@/components/ScreenHeader";
import { Composer } from "@/features/chat/components/Composer";
import { PromptChips } from "@/features/chat/components/PromptChips";
import { Hero } from "@/features/home/components/Hero";
import { NeedsYouCard } from "@/features/home/components/NeedsYouCard";
import { NowNextCard } from "@/features/home/components/NowNextCard";
import { useHomeScreen } from "@/features/home/useHomeScreen";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function HomeScreen() {
  const home = useHomeScreen();
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  function handleSend(text: string) {
    void home.submit(text);
  }

  function handleAttach() {
    void home.files.attach();
  }

  function handleRefresh() {
    void home.refresh();
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader />
      <KeyboardAvoidingView
        behavior="padding"
        style={styles.body}
      >
        <ScrollView
          style={styles.body}
          contentContainerStyle={[
            styles.content,
            !home.hasCards && styles.centred,
            { paddingBottom: insets.bottom + spacing.lg },
          ]}
          keyboardShouldPersistTaps="handled"
          refreshControl={
            <RefreshControl
              refreshing={home.isRefreshing}
              onRefresh={handleRefresh}
              tintColor={colors.textTertiary}
            />
          }
        >
          <Hero
            name={home.name}
            experts={home.experts}
            selected={home.selected}
            onSelect={home.select}
          />
          <View style={styles.composer}>
            <Composer
              layout="card"
              onSend={home.submit}
              isDisabled={home.isStarting}
              placeholder={home.placeholder}
              onAttach={handleAttach}
              attachments={home.files.attachments}
              onRemoveAttachment={home.files.remove}
              onRetryAttachment={home.files.retry}
            />
          </View>
          {home.error || home.files.message ? (
            <Text accessibilityLiveRegion="polite" style={styles.error}>
              {home.error ?? home.files.message}
            </Text>
          ) : null}
          <PromptChips
            key={home.promptExpert?.name ?? "otto"}
            prompts={home.prompts}
            expert={home.promptExpert}
            onPick={handleSend}
          />
          {home.hasCards ? (
            <View style={styles.cards}>
              <NeedsYouCard items={home.attention} />
              <NowNextCard active={home.activeTask} upcoming={home.upcomingTask} />
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  body: { flex: 1 },
  content: { flexGrow: 1, gap: spacing.lg, paddingTop: spacing.xl },
  // With nothing below, the hero and composer sit in the middle, as on the web.
  centred: { justifyContent: "center", paddingTop: 0 },
  composer: { marginHorizontal: spacing.lg },
  cards: { gap: spacing.md, paddingHorizontal: spacing.lg, paddingTop: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
  },
}));
