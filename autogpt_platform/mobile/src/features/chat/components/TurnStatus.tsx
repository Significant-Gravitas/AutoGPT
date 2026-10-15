import { ActivityIndicator, StyleSheet, Text, View } from "react-native";

import { PlainButton } from "../../../design/components/Buttons";
import { colors, spacing, type } from "../../../design/theme";
import type { TurnPhase } from "../conversationController";

interface Props {
  turn: TurnPhase;
  statusText: string | null;
  notice: string | null;
  participantName: string;
  onRetry: () => void;
}

/** Sits under the newest message: what the reply is doing, or what went wrong. */
export function TurnStatus({
  turn,
  statusText,
  notice,
  participantName,
  onRetry,
}: Props) {
  const label = phaseLabel(turn, statusText, participantName);
  if (!label && !notice) return <View style={styles.spacer} />;
  const offline = turn === "offline";
  return (
    <View style={styles.container} accessibilityLiveRegion="polite">
      {label ? (
        <View style={styles.line}>
          {offline ? null : <ActivityIndicator size="small" />}
          <Text style={[type.footnote, styles.text]} testID="turn-status">
            {label}
          </Text>
        </View>
      ) : null}
      {notice ? (
        <Text style={[type.footnote, styles.notice]} accessibilityRole="alert">
          {notice}
        </Text>
      ) : null}
      {offline || (notice && turn === "idle") ? (
        <PlainButton
          title={offline ? "Reconnect" : "Try Again"}
          onPress={onRetry}
        />
      ) : null}
    </View>
  );
}

function phaseLabel(
  turn: TurnPhase,
  statusText: string | null,
  name: string,
): string | null {
  switch (turn) {
    case "idle":
      return null;
    case "sending":
      return "Sending…";
    case "streaming":
      return statusText ?? `${name} is working…`;
    case "reconnecting":
      return "Reconnecting…";
    case "stopping":
      return "Stopping…";
    case "queued":
      return "Waiting for a free slot…";
    case "offline":
      return "Not connected. The reply keeps going on the server.";
  }
}

const styles = StyleSheet.create({
  spacer: { height: spacing.md },
  container: {
    alignItems: "center",
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  line: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  text: { color: colors.secondaryLabel },
  notice: { color: colors.danger, textAlign: "center" },
});
