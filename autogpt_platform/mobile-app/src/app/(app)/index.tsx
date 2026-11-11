import { Image } from "expo-image";
import { KeyboardAvoidingView, Platform, Text, View } from "react-native";

import { ScreenHeader } from "@/components/ScreenHeader";
import { Composer } from "@/features/chat/components/Composer";
import { PromptChips } from "@/features/chat/components/PromptChips";
import { useHomeScreen } from "@/features/home/useHomeScreen";
import { createStyles, spacing, type } from "@/lib/theme";

const appIcon = require("../../../assets/images/icon.png");

export default function HomeScreen() {
  const { prompts, isStarting, error, submit } = useHomeScreen();
  const styles = useStyles();

  function handleSend(text: string) {
    void submit(text);
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader />
      <View style={styles.middle}>
        <Image
          source={appIcon}
          style={styles.icon}
          accessibilityIgnoresInvertColors
        />
        <Text style={styles.heading}>What should we work on?</Text>
      </View>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.bottom}
      >
        {error ? (
          <Text accessibilityLiveRegion="polite" style={styles.error}>
            {error}
          </Text>
        ) : null}
        <PromptChips prompts={prompts} onPick={handleSend} />
        <Composer onSend={submit} isDisabled={isStarting} />
      </KeyboardAvoidingView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  middle: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.lg,
    paddingHorizontal: spacing.xl,
  },
  icon: { width: 56, height: 56, borderRadius: 14 },
  heading: { ...type.title, color: colors.text, textAlign: "center" },
  bottom: { gap: spacing.sm },
  error: {
    ...type.small,
    color: colors.error,
    paddingHorizontal: spacing.lg,
  },
}));
