import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { StatusBar } from "expo-status-bar";
import { useEffect } from "react";
import { useColorScheme } from "react-native";
import { KeyboardProvider } from "react-native-keyboard-controller";

import { SessionProvider, useSession } from "../features/auth/SessionProvider";

void SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const scheme = useColorScheme();
  return (
    <ThemeProvider value={scheme === "dark" ? DarkTheme : DefaultTheme}>
      <KeyboardProvider>
        <SessionProvider>
          <RootNavigator />
        </SessionProvider>
      </KeyboardProvider>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}

/**
 * The splash screen stays up while the saved sign-in is read, so neither
 * group flashes before the right one is known.
 */
function RootNavigator() {
  const { state } = useSession();
  const restoring = state.status === "restoring";

  useEffect(() => {
    if (!restoring) SplashScreen.hide();
  }, [restoring]);

  if (restoring) return null;
  const signedIn = state.status === "signedIn";
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Protected guard={!signedIn}>
        <Stack.Screen name="(auth)" />
      </Stack.Protected>
      <Stack.Protected guard={signedIn}>
        <Stack.Screen name="(app)" />
      </Stack.Protected>
    </Stack>
  );
}
