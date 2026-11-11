import {
  Folder01Icon,
  PencilEdit02Icon,
  SidebarRight01Icon,
} from "@hugeicons/core-free-icons";
import { useLocalSearchParams } from "expo-router";
import { ActivityIndicator, Text, View } from "react-native";
import { Drawer } from "react-native-drawer-layout";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { ChatIdentity } from "@/features/chat/components/ChatIdentity";
import { ChatThread } from "@/features/chat/components/ChatThread";
import { useChatScreen } from "@/features/chat/useChatScreen";
import { ExpertPanel } from "@/features/expertPanel/components/ExpertPanel";
import { FilesPanel } from "@/features/files/components/FilesPanel";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function ChatScreen() {
  const { sessionId } = useLocalSearchParams<{ sessionId: string }>();
  const {
    sessionID,
    detail,
    isPending,
    refetch,
    expertID,
    identity,
    newChat,
    isFilesOpen,
    openFiles,
    closeFiles,
  } = useChatScreen(sessionId);
  const styles = useStyles();
  const colors = useColors();
  const hasFiles = sessionID !== null && detail !== undefined;

  function handleRetry() {
    void refetch();
  }

  function renderBody() {
    // Data first: a failed background refetch must not unmount a live thread.
    if (sessionID && detail) {
      return <ChatThread key={sessionID} sessionID={sessionID} detail={detail} />;
    }
    if (isPending) {
      return (
        <View style={styles.center}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      );
    }
    return (
      <View style={styles.center}>
        <Text style={styles.notice}>{"Couldn't open this chat."}</Text>
        {sessionID ? (
          <Button
            label="Try again"
            onPress={handleRetry}
            variant="secondary"
            size="small"
          />
        ) : null}
      </View>
    );
  }

  function renderPanel() {
    if (!sessionID) return null;
    if (expertID) {
      return (
        <ExpertPanel
          sessionID={sessionID}
          expertID={expertID}
          name={identity?.name ?? "Expert"}
          role={identity?.role ?? null}
          avatar={identity?.avatar ?? null}
          tint={identity?.tint ?? null}
          isOpen={isFilesOpen}
          onClose={closeFiles}
        />
      );
    }
    return (
      <FilesPanel sessionID={sessionID} isOpen={isFilesOpen} onClose={closeFiles} />
    );
  }

  return (
    <Drawer
      open={isFilesOpen}
      onOpen={openFiles}
      onClose={closeFiles}
      drawerPosition="right"
      drawerType="front"
      // The app's menu drawer owns the swipe; this panel opens from its button.
      swipeEnabled={false}
      overlayAccessibilityLabel={expertID ? "Close details" : "Close files"}
      drawerStyle={styles.panel}
      renderDrawerContent={renderPanel}
    >
      <View style={styles.screen}>
        <ScreenHeader
          center={identity ? <ChatIdentity {...identity} /> : null}
          right={
            <>
              {hasFiles ? (
                <IconButton
                  icon={expertID ? SidebarRight01Icon : Folder01Icon}
                  label={expertID ? "Details" : "Files"}
                  onPress={openFiles}
                />
              ) : null}
              <IconButton icon={PencilEdit02Icon} label="New chat" onPress={newChat} />
            </>
          }
        />
        {renderBody()}
      </View>
    </Drawer>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  panel: { width: "100%", backgroundColor: colors.background },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
    padding: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
