import * as Clipboard from "expo-clipboard";
import { ActionSheetIOS, Alert, Platform, Share } from "react-native";

import { tapMenu } from "@/lib/haptics";

type MessageAction = "copy" | "share";

/** The long-press menu on a message: copy or share its words. */
export async function openMessageActions(text: string) {
  if (!text) return;
  tapMenu();
  const action = await chooseAction();
  if (action === "copy") await Clipboard.setStringAsync(text);
  if (action === "share") await Share.share({ message: text }).catch(() => undefined);
}

function chooseAction() {
  return new Promise<MessageAction | null>((resolve) => {
    if (Platform.OS === "ios") {
      ActionSheetIOS.showActionSheetWithOptions(
        { options: ["Copy", "Share", "Cancel"], cancelButtonIndex: 2 },
        (index) => resolve(index === 0 ? "copy" : index === 1 ? "share" : null),
      );
      return;
    }
    Alert.alert(
      "Message",
      undefined,
      [
        { text: "Cancel", style: "cancel", onPress: () => resolve(null) },
        { text: "Share", onPress: () => resolve("share") },
        { text: "Copy", onPress: () => resolve("copy") },
      ],
      { cancelable: true, onDismiss: () => resolve(null) },
    );
  });
}
