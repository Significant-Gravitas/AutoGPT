import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { WebView } from "react-native-webview";

import { ScreenHeader } from "@/components/ScreenHeader";
import { useSettingsScreen } from "@/features/settings/useSettingsScreen";
import { colors, radius, spacing, type } from "@/lib/theme";

export default function SettingsScreen() {
  const {
    settingsURL,
    cookie,
    script,
    isError,
    isBlocked,
    webViewKey,
    retry,
    webViewRef,
    handleNavigation,
    handleStateChange,
  } = useSettingsScreen();

  function renderNotice(message: string) {
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>{message}</Text>
        <Pressable
          accessibilityRole="button"
          onPress={retry}
          style={({ pressed }) => [styles.button, pressed && styles.pressed]}
        >
          <Text style={styles.buttonText}>Try again</Text>
        </Pressable>
      </View>
    );
  }

  function renderBody() {
    if (isError) {
      return renderNotice("Couldn't open Settings. Try again.");
    }
    if (isBlocked) {
      return renderNotice(
        "Settings didn't open here. Try again, or open Settings on the web.",
      );
    }
    if (cookie === undefined) {
      return (
        <View style={styles.center}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      );
    }
    return (
      <WebView
        key={webViewKey}
        ref={webViewRef}
        source={{ uri: settingsURL, headers: { Cookie: cookie } }}
        injectedJavaScriptBeforeContentLoaded={script}
        sharedCookiesEnabled
        originWhitelist={["*"]}
        onShouldStartLoadWithRequest={handleNavigation}
        onNavigationStateChange={handleStateChange}
        style={styles.webview}
      />
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader title="Settings" />
      {renderBody()}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  webview: { flex: 1, backgroundColor: colors.background },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.lg,
    paddingHorizontal: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  button: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: radius.md,
    backgroundColor: colors.surfaceRaised,
  },
  pressed: { opacity: 0.8 },
  buttonText: { ...type.label, color: colors.text },
});
