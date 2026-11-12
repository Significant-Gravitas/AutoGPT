import "@/polyfills";

import { PersistQueryClientProvider } from "@tanstack/react-query-persist-client";
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
import { KeyboardProvider } from "react-native-keyboard-controller";

import { applyAppearance, readAppearance } from "@/lib/appearance";
import { authClient } from "@/lib/auth/client";
import {
  PERSIST_BUSTER,
  PERSIST_MAX_AGE_MS,
  queryPersister,
  shouldPersistQuery,
} from "@/lib/queryPersist";
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

const persistOptions = {
  persister: queryPersister,
  maxAge: PERSIST_MAX_AGE_MS,
  buster: PERSIST_BUSTER,
  dehydrateOptions: { shouldDehydrateQuery: shouldPersistQuery },
};

export default function RootLayout() {
  const styles = useStyles();
  const scheme = useScheme();
  const colors = useColors();
  return (
    <GestureHandlerRootView style={styles.root}>
      <KeyboardProvider>
        <PersistQueryClientProvider client={queryClient} persistOptions={persistOptions}>
          <ThemeProvider value={navigationTheme(scheme, colors)}>
            <StatusBar style="auto" />
            <RootStack />
          </ThemeProvider>
        </PersistQueryClientProvider>
      </KeyboardProvider>
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
