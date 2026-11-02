import { Stack } from "expo-router";

import { colors } from "../../../../design/theme";

export default function ChatsLayout() {
  return (
    <Stack screenOptions={{ headerTintColor: colors.tint }}>
      <Stack.Screen
        name="index"
        options={{ title: "Chats", headerLargeTitleEnabled: true }}
      />
    </Stack>
  );
}
