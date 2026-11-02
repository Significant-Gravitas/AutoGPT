import { Stack } from "expo-router";

import { colors } from "../../../../design/theme";

export default function CompanyLayout() {
  return (
    <Stack screenOptions={{ headerTintColor: colors.tint }}>
      <Stack.Screen
        name="index"
        options={{
          title: "Company",
          headerLargeTitleEnabled: true,
          contentStyle: { backgroundColor: colors.groupedBackground },
        }}
      />
    </Stack>
  );
}
