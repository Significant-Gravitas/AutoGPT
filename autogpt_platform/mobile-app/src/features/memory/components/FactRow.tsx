import { type AccessibilityActionEvent, Pressable, Text } from "react-native";

import type { MemoryFact } from "@/lib/api/memory";
import { createStyles, spacing, type } from "@/lib/theme";
import { shortTime } from "@/lib/time";

import { factText } from "../helpers";

interface Props {
  fact: MemoryFact;
  onLongPress: (fact: MemoryFact) => void;
  /** Offered to screen readers, which can't reach the long-press menu. */
  onForget: (uuid: string) => void;
}

export function FactRow({ fact, onLongPress, onForget }: Props) {
  const styles = useStyles();
  const text = factText(fact);
  const date = fact.created_at ? shortTime(fact.created_at) : null;

  function handleLongPress() {
    onLongPress(fact);
  }

  function handleAccessibilityAction(event: AccessibilityActionEvent) {
    if (event.nativeEvent.actionName === "forget") onForget(fact.uuid);
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={[text, date].filter(Boolean).join(", ")}
      accessibilityHint="Hold for options"
      accessibilityActions={[{ name: "forget", label: "Forget" }]}
      onAccessibilityAction={handleAccessibilityAction}
      onLongPress={handleLongPress}
      delayLongPress={350}
      style={({ pressed }) => [styles.row, pressed && styles.pressed]}
    >
      <Text style={styles.text}>{text}</Text>
      {date ? <Text style={styles.date}>{date}</Text> : null}
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: spacing.md,
    minHeight: 52,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  pressed: { backgroundColor: colors.surfacePressed },
  text: { ...type.body, flex: 1, color: colors.text },
  date: { ...type.small, paddingTop: 3, color: colors.textTertiary },
}));
