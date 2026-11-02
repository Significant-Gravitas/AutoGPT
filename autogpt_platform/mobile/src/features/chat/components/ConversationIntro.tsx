import { Keyboard, Pressable, StyleSheet, Text, View } from "react-native";

import { Avatar } from "../../../design/components/Avatar";
import { colors, spacing, type } from "../../../design/theme";
import type { Participant } from "../../company/participants";

/** The empty state of a conversation that has no messages yet. */
export function ConversationIntro({
  participant,
}: {
  participant: Participant;
}) {
  return (
    <Pressable
      style={styles.container}
      onPress={Keyboard.dismiss}
      accessible={false}
    >
      <View style={styles.content}>
        <Avatar name={participant.name} uri={participant.avatarURL} size={72} />
        <Text style={[type.title2, styles.name]} accessibilityRole="header">
          {participant.name}
        </Text>
        {participant.role ? (
          <Text style={[type.subheadline, styles.role]}>
            {participant.role}
          </Text>
        ) : null}
        <Text style={[type.body, styles.hint]}>
          {participant.readOnlyReason ??
            `Your first message starts a new conversation with ${participant.name}.`}
        </Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    padding: spacing.xxl,
  },
  name: { color: colors.label, marginTop: spacing.sm },
  role: { color: colors.secondaryLabel },
  hint: {
    color: colors.secondaryLabel,
    textAlign: "center",
    marginTop: spacing.md,
  },
});
