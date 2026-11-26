import { Drawer } from "expo-router/drawer";
import type { ReactNode } from "react";
import { View } from "react-native";

import { PageFrame } from "@/components/PageFrame";
import { AppDrawerContent } from "@/features/drawer/AppDrawerContent";
import { createStyles } from "@/lib/theme";

export default function AppLayout() {
  const styles = useStyles();
  return (
    // The menu's grey, behind the see-through drawer and the page's rounded corners.
    <View style={styles.root}>
      <Drawer
        drawerContent={(props) => <AppDrawerContent {...props} />}
        screenLayout={renderPage}
        screenOptions={{
          headerShown: false,
          // Pushes the page aside and leaves a strip of it to tap closed.
          drawerType: "slide",
          // See-through, so the page's shadow falls on the menu.
          drawerStyle: styles.drawer,
          sceneStyle: styles.scene,
          // The wash lives in PageFrame, where it shares the page's corners;
          // this overlay only takes the tap that closes the menu.
          overlayColor: "transparent",
          overlayAccessibilityLabel: "Close menu",
        }}
      >
        <Drawer.Screen name="index" />
        <Drawer.Screen name="chat/[sessionId]" />
        <Drawer.Screen name="chats" />
        <Drawer.Screen name="team" />
        <Drawer.Screen name="create-expert" />
        <Drawer.Screen name="marketplace/index" />
        <Drawer.Screen name="marketplace/[templateId]" />
        <Drawer.Screen name="files/index" />
        <Drawer.Screen name="files/[folderId]" />
        <Drawer.Screen name="settings" />
        <Drawer.Screen name="plan" />
      </Drawer>
    </View>
  );
}

function renderPage({ children }: { children: ReactNode }) {
  return <PageFrame>{children}</PageFrame>;
}

const useStyles = createStyles((colors) => ({
  root: { flex: 1, backgroundColor: colors.menuBackground },
  drawer: { width: "78%", backgroundColor: "transparent" },
  scene: { backgroundColor: "transparent" },
}));
