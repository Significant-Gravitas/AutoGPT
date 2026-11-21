import { RefreshControl, ScrollView, Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { ScreenHeader } from "@/components/ScreenHeader";
import { Composer } from "@/features/chat/components/Composer";
import { HomeRows } from "@/features/home/components/HomeRows";
import { RecipientPicker } from "@/features/home/components/RecipientPicker";
import { useHomeScreen } from "@/features/home/useHomeScreen";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function HomeScreen() {
  const home = useHomeScreen();
  const styles = useStyles();
  const colors = useColors();

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
      <ScreenHeader
        menuVariant="icon"
        center={
          <RecipientPicker
            experts={home.experts}
            selected={home.selected}
            onSelect={home.select}
          />
        }
      />
      <KeyboardAvoidingView behavior="padding" style={styles.body}>
        <ScrollView
          style={styles.body}
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
          <HomeRows
            key={home.selected?.id ?? "otto"}
            rows={home.rows}
            onPick={handleSend}
          />
        </ScrollView>
        {home.error || home.files.message ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {home.error ?? home.files.message}
          </Text>
        ) : null}
        <Composer
          onSend={home.submit}
          isDisabled={home.isStarting}
          placeholder={home.placeholder}
          onAttach={handleAttach}
          attachments={home.files.attachments}
          onRemoveAttachment={home.files.remove}
          onRetryAttachment={home.files.retry}
        />
      </KeyboardAvoidingView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  body: { flex: 1 },
  content: {
    flexGrow: 1,
    justifyContent: "flex-end",
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.md,
  },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.sm,
  },
}));
