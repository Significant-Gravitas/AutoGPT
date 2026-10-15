import { Stack } from "expo-router";

import { colors } from "../../design/theme";
import { AppDataProvider } from "../../features/app/AppDataProvider";
import { useSignedIn } from "../../features/auth/SessionProvider";

export default function SignedInLayout() {
  const { account, server } = useSignedIn();
  return (
    <AppDataProvider key={`${server.id}:${account.key}`}>
      <Stack
        screenOptions={{
          headerTintColor: colors.tint,
          headerBackButtonDisplayMode: "minimal",
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="conversation/[sessionId]"
          options={{
            title: "",
            contentStyle: { backgroundColor: colors.background },
          }}
        />
        <Stack.Screen
          name="account"
          options={{
            presentation: "modal",
            title: "Account",
            contentStyle: { backgroundColor: colors.groupedBackground },
          }}
        />
        <Stack.Screen
          name="tool-details"
          options={{
            presentation: "formSheet",
            headerShown: false,
            sheetAllowedDetents: [0.55, 1],
            sheetGrabberVisible: true,
            contentStyle: { backgroundColor: colors.groupedBackground },
          }}
        />
      </Stack>
    </AppDataProvider>
  );
}
