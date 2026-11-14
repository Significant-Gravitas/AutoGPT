import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { ShareChatButton } from "@/features/share/components/ShareChatButton";
import { createStyles } from "@/lib/theme";

interface Props {
  sessionID: string;
  onClose: () => void;
}

/** The chat panel's own header: back to the chat, and share. */
export function PanelTopBar({ sessionID, onClose }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.bar}>
      <IconButton icon={ArrowLeft01Icon} label="Back to chat" onPress={onClose} />
      <ShareChatButton sessionID={sessionID} />
    </View>
  );
}

const useStyles = createStyles(() => ({
  bar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
}));
