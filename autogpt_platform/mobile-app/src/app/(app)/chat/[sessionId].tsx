import { Folder01Icon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useLocalSearchParams } from "expo-router";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { Drawer } from "react-native-drawer-layout";

import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { ChatIdentity } from "@/features/chat/components/ChatIdentity";
import { ChatThread } from "@/features/chat/components/ChatThread";
import { useChatScreen } from "@/features/chat/useChatScreen";
import { FilesPanel } from "@/features/files/components/FilesPanel";
import { colors, radius, spacing, type } from "@/lib/theme";

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
          <Pressable
            accessibilityRole="button"
            onPress={handleRetry}
            style={({ pressed }) => [styles.retry, pressed && styles.pressed]}
          >
            <Text style={styles.retryLabel}>Try again</Text>
          </Pressable>
        ) : null}
      </View>
    );
  }

  function renderFilesPanel() {
    if (!sessionID) return null;
    return (
      <FilesPanel
        sessionID={sessionID}
        expertID={expertID}
        expertName={expertID ? (identity?.name ?? null) : null}
        isOpen={isFilesOpen}
      />
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
      overlayAccessibilityLabel="Close files"
      drawerStyle={styles.panel}
      renderDrawerContent={renderFilesPanel}
    >
      <View style={styles.screen}>
        <ScreenHeader
          center={identity ? <ChatIdentity {...identity} /> : null}
          right={
            <>
              {hasFiles ? (
                <IconButton icon={Folder01Icon} label="Files" onPress={openFiles} />
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

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  panel: { width: "84%", backgroundColor: colors.background },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.md,
    padding: spacing.xl,
  },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  retry: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceRaised,
  },
  pressed: { opacity: 0.6 },
  retryLabel: { ...type.label, color: colors.text },
});
