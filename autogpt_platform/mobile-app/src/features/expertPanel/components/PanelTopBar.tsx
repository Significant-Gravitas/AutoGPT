import { ArrowLeft01Icon, Settings02Icon } from "@hugeicons/core-free-icons";
import { router } from "expo-router";
import { View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { ShareChatButton } from "@/features/share/components/ShareChatButton";
import { createStyles, spacing } from "@/lib/theme";

interface Props {
  sessionID: string;
  /** Set in an expert's chat: shows the button that opens its settings. */
  expertID: string | null;
  expertName?: string;
  onClose: () => void;
}

/** The chat panel's own header: back to the chat, manage the expert, and share. */
export function PanelTopBar({ sessionID, expertID, expertName, onClose }: Props) {
  const styles = useStyles();

  function openManage() {
    if (!expertID) return;
    onClose();
    router.push(`/expert/${expertID}?from=${encodeURIComponent(sessionID)}`);
  }

  return (
    <View style={styles.bar}>
      <IconButton icon={ArrowLeft01Icon} label="Back to chat" onPress={onClose} variant="floating" />
      <View style={styles.right}>
        {expertID ? (
          <IconButton
            icon={Settings02Icon}
            label={`Manage ${expertName ?? "expert"}`}
            onPress={openManage}
            variant="floating"
          />
        ) : null}
        <ShareChatButton sessionID={sessionID} />
      </View>
    </View>
  );
}

const useStyles = createStyles(() => ({
  bar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  right: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
}));
