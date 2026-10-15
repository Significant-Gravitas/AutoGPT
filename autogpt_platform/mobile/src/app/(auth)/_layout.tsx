import { Stack } from "expo-router";

import { colors } from "../../design/theme";

export default function AuthLayout() {
  return (
    <Stack
      screenOptions={{
        headerTintColor: colors.tint,
        headerBackButtonDisplayMode: "minimal",
        contentStyle: { backgroundColor: colors.groupedBackground },
      }}
    >
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="own-server" options={{ title: "Your Server" }} />
      <Stack.Screen name="sign-in" options={{ title: "Sign In" }} />
    </Stack>
  );
}
