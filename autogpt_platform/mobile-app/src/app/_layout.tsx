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
import { useState } from "react";
import { View } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";

import { applyAppearance, readAppearance } from "@/lib/appearance";
import { authClient } from "@/lib/auth/client";
import { queryClient } from "@/lib/session";
import { createStyles, useColors, useScheme, type Colors, type Scheme } from "@/lib/theme";

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
  return (
    <GestureHandlerRootView style={styles.root}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider value={navigationTheme(scheme, colors)}>
          <StatusBar style="auto" />
          <RootStack />
        </ThemeProvider>
      </QueryClientProvider>
    </GestureHandlerRootView>
  );
}

function RootStack() {
  const styles = useStyles();
  const { data: session, isPending } = authClient.useSession();
  const [hasSettled, setHasSettled] = useState(false);

  // Every refetch (e.g. on app focus) reports isPending again while signed
  // out; only the first one may blank the screen, or typed login text is lost.
  if (!isPending && !hasSettled) setHasSettled(true);
  if (!hasSettled) return <View style={styles.root} />;

  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Protected guard={!!session}>
        <Stack.Screen name="(app)" />
      </Stack.Protected>
      <Stack.Protected guard={!session}>
        <Stack.Screen name="login" />
      </Stack.Protected>
    </Stack>
  );
}

const useStyles = createStyles((colors) => ({
  root: { flex: 1, backgroundColor: colors.background },
}));
