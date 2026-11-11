import { Text, View } from "react-native";

import type { Expert } from "@/lib/api/experts";
import { createStyles, spacing } from "@/lib/theme";

import { greetingLine, introParts, sentenceWords } from "../helpers";
import { RecipientPicker } from "./RecipientPicker";

interface Props {
  name: string | null;
  experts: Expert[];
  /** null is Otto. */
  selected: Expert | null;
  onSelect: (expertID: string | null) => void;
}

// Each word is its own flex item so the sentence wraps around the picker;
// a View nested in Text misaligns on Android.
export function Hero({ name, experts, selected, onSelect }: Props) {
  const styles = useStyles();
  const { before, after } = introParts(selected);
  const tail = sentenceWords(after);

  return (
    <View style={styles.hero}>
      <Text accessibilityRole="header" style={styles.greeting}>
        {greetingLine(name)}
      </Text>
      <View style={styles.sentence}>
        <Words words={sentenceWords(before).words} label={before} />
        <View style={styles.picker}>
          <RecipientPicker experts={experts} selected={selected} onSelect={onSelect} />
          {tail.mark ? (
            <Text
              style={styles.word}
              accessibilityElementsHidden
              importantForAccessibility="no-hide-descendants"
            >
              {tail.mark}
            </Text>
          ) : null}
        </View>
        <Words words={tail.words} label={after} />
      </View>
    </View>
  );
}

interface WordsProps {
  words: string[];
  label: string;
}

// The first word speaks the whole part, so a screen reader does not stop on every word.
function Words({ words, label }: WordsProps) {
  const styles = useStyles();
  return words.map((word, index) => (
    <Text
      key={`${index}-${word}`}
      style={styles.word}
      accessibilityLabel={index === 0 ? label.replace(/^[,.]?\s*/, "") : undefined}
      accessibilityElementsHidden={index > 0}
      importantForAccessibility={index > 0 ? "no-hide-descendants" : "auto"}
    >
      {word}
    </Text>
  ));
}

const useStyles = createStyles((colors) => ({
  hero: { gap: spacing.sm, paddingHorizontal: spacing.lg },
  greeting: {
    fontSize: 24,
    fontWeight: "600",
    lineHeight: 30,
    textAlign: "center",
    color: colors.text,
  },
  sentence: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    alignItems: "center",
    columnGap: 5,
    rowGap: 2,
  },
  picker: { flexDirection: "row", alignItems: "center", gap: 2 },
  word: { fontSize: 20, lineHeight: 30, textAlign: "center", color: colors.text },
}));
