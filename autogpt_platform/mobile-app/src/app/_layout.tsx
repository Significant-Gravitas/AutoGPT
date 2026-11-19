import "@/polyfills";

import { QueryClientProvider } from "@tanstack/react-query";
import {
  DarkTheme,
  DefaultTheme,
  Stack,
  ThemeProvider,
  type Theme,
} from "expo-router";
import { StatusBar } from "expo-status-bar";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { KeyboardProvider } from "react-native-keyboard-controller";

import { applyAppearance, readAppearance } from "@/lib/appearance";
import { useSession } from "@/lib/auth/nativeSession";
import { CachePersistence } from "@/components/CachePersistence";
import { queryClient, signOutIfSessionEnded } from "@/lib/session";
import { createStyles, useColors, useScheme, type Colors, type Scheme } from "@/lib/theme";
import { useMountEffect } from "@/lib/useMountEffect";

// Before the first render, so the app opens in the scheme picked last time.
applyAppearance(readAppearance());

function navigationTheme(scheme: Scheme, colors: Colors): Theme {
  const base = scheme === "light" ? DefaultTheme : DarkTheme;
  return {
    ...base,
    colors: {
      ...base.colors,
      primary: colors.brand,
      background: colors.background,
      card: colors.surface,
      text: colors.text,
      border: colors.border,
      notification: colors.error,
    },
  };
}

export default function RootLayout() {
  const styles = useStyles();
  const scheme = useScheme();
  const colors = useColors();
  useMountEffect(() => {
    void signOutIfSessionEnded();
  });
  return (
    <GestureHandlerRootView style={styles.root}>
      <KeyboardProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider value={navigationTheme(scheme, colors)}>
            <StatusBar style="auto" />
            <RootStack />
          </ThemeProvider>
        </QueryClientProvider>
      </KeyboardProvider>
    </GestureHandlerRootView>
  );
}

function RootStack() {
  // Read from the phone, so the first render already knows who is signed in.
  const session = useSession();
  const userID = session?.user.id;
  return (
    <>
      {/* Keyed by user: a new account starts from its own saved copy, or none. */}
      {userID ? <CachePersistence key={userID} userID={userID} /> : null}
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Protected guard={!!session}>
          <Stack.Screen name="(app)" />
          {/* A sign-in in the system browser comes back here. */}
          <Stack.Screen name="integrations/oauth" />
        </Stack.Protected>
        <Stack.Protected guard={!session}>
          <Stack.Screen name="login" />
        </Stack.Protected>
      </Stack>
    </>
  );
}

const useStyles = createStyles((colors) => ({
  root: { flex: 1, backgroundColor: colors.background },
}));
