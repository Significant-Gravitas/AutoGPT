import { InformationCircleIcon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useLocalSearchParams } from "expo-router";
import { ActivityIndicator, Text, View } from "react-native";
import { Drawer } from "react-native-drawer-layout";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { PageFrame } from "@/components/PageFrame";
import { FloatingHeader } from "@/components/FloatingHeader";
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
    isCached,
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
      return (
        <ChatThread
          key={sessionID}
          sessionID={sessionID}
          detail={detail}
          isCached={isCached}
        />
      );
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

  function renderIdentity() {
    if (identity) {
      return (
        <ChatIdentity
          name={identity.name}
          avatar={identity.avatar}
          tint={identity.tint}
          onPress={hasFiles ? openFiles : undefined}
          actionLabel={expertID ? "open details" : "open files"}
        />
      );
    }
    // The expert's name is still loading; the panel opens anyway.
    if (!hasFiles) return null;
    return (
      <IconButton
        icon={InformationCircleIcon}
        label={expertID ? "Open details" : "Open files"}
        onPress={openFiles}
        variant="floating"
      />
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
      // Pushes the chat aside, as the menu does from the other side.
      drawerType="slide"
      // The app's menu drawer owns the swipe; this panel opens from the name pill.
      swipeEnabled={false}
      overlayAccessibilityLabel={expertID ? "Close details" : "Close files"}
      // PageFrame draws the wash; this overlay only takes the closing tap.
      overlayStyle={styles.overlay}
      drawerStyle={styles.panel}
      style={styles.backdrop}
      renderDrawerContent={renderPanel}
    >
      <PageFrame>
        <View style={styles.screen}>
          {renderBody()}
          {/* After the body, so it floats over the thread. */}
          <FloatingHeader
            center={renderIdentity()}
            right={
              <IconButton
                icon={PencilEdit02Icon}
                label="New chat"
                onPress={newChat}
                variant="floating"
              />
            }
          />
        </View>
      </PageFrame>
    </Drawer>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  // See-through, so the chat's shadow falls on the panel.
  panel: { width: "85%", backgroundColor: "transparent" },
  overlay: { backgroundColor: "transparent" },
  backdrop: { backgroundColor: colors.panelBackground },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
    padding: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
