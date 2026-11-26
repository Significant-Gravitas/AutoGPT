import * as Clipboard from "expo-clipboard";
import { Share } from "react-native";

import { chooseAction } from "@/lib/actionMenu";
import { tapMenu } from "@/lib/haptics";

/** The long-press menu on a message: copy or share its words. */
export async function openMessageActions(text: string) {
  if (!text) return;
  tapMenu();
  const action = await chooseAction("Message", [
    { id: "copy", label: "Copy" },
    { id: "share", label: "Share" },
  ]);
  if (action === "copy") await Clipboard.setStringAsync(text);
  if (action === "share") await Share.share({ message: text }).catch(() => undefined);
}
