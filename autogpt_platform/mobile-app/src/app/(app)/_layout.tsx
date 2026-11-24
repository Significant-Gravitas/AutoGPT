import { Drawer } from "expo-router/drawer";

import { AppDrawerContent } from "@/features/drawer/AppDrawerContent";
import { useColors } from "@/lib/theme";

export default function AppLayout() {
  const colors = useColors();
  return (
    <Drawer
      drawerContent={(props) => <AppDrawerContent {...props} />}
      screenOptions={{
        headerShown: false,
        // Full width, pushing the page aside as it opens.
        drawerType: "slide",
        drawerStyle: { width: "100%", backgroundColor: colors.background },
        overlayColor: "transparent",
      }}
    >
      <Drawer.Screen name="index" />
      <Drawer.Screen name="chat/[sessionId]" />
      <Drawer.Screen name="chats" />
      <Drawer.Screen name="team" />
      <Drawer.Screen name="create-expert" />
      <Drawer.Screen name="settings" />
    </Drawer>
  );
}
