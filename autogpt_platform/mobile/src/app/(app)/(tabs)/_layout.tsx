import { NativeTabs } from "expo-router/unstable-native-tabs";

import { colors } from "../../../design/theme";

/** The two primary destinations: conversations, and the AI company you talk to. */
export default function TabsLayout() {
  return (
    <NativeTabs tintColor={colors.tint}>
      <NativeTabs.Trigger name="chats">
        <NativeTabs.Trigger.Icon
          sf={{
            default: "bubble.left.and.bubble.right",
            selected: "bubble.left.and.bubble.right.fill",
          }}
          md="forum"
        />
        <NativeTabs.Trigger.Label>Chats</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="company">
        <NativeTabs.Trigger.Icon
          sf={{ default: "person.3", selected: "person.3.fill" }}
          md="groups"
        />
        <NativeTabs.Trigger.Label>Company</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}
