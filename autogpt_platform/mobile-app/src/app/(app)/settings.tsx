import { ActivityIndicator, Platform, Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { WebView } from "react-native-webview";

import { Button } from "@/components/Button";
import { ScreenHeader } from "@/components/ScreenHeader";
import { useSettingsScreen } from "@/features/settings/useSettingsScreen";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

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
  const styles = useStyles();
  const colors = useColors();

  function renderNotice(message: string) {
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>{message}</Text>
        <Button label="Try again" onPress={retry} variant="secondary" size="small" />
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
      {/* iOS scrolls a focused field into view itself; Android needs the room made. */}
      <KeyboardAvoidingView
        behavior="padding"
        enabled={Platform.OS === "android"}
        style={styles.body}
      >
        {renderBody()}
      </KeyboardAvoidingView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  body: { flex: 1 },
  webview: { flex: 1, backgroundColor: colors.background },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.lg,
    paddingHorizontal: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
