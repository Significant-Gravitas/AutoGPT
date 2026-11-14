import {
  CheckmarkCircle02Icon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import type { SentAnswer } from "../helpers";

interface Props {
  answers: SentAnswer[];
}

/** A card the chat has moved past: what was asked and answered, nothing to tap. */
export function AnsweredOnboarding({ answers }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.answered}>
      <View style={styles.header}>
        <Icon icon={CheckmarkCircle02Icon} size={16} color={colors.textTertiary} />
        <Text style={styles.headerText}>Setup questions</Text>
      </View>
      {answers.length > 0 ? (
        <View style={styles.list}>
          {answers.map((item) => (
            <View key={item.keyword} style={styles.item}>
              <Icon icon={MessageQuestionIcon} size={14} color={colors.textTertiary} />
              <View style={styles.itemText}>
                <Text style={styles.question}>{item.question}</Text>
                <Text style={styles.answer}>{item.answer}</Text>
              </View>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  answered: { gap: spacing.xs },
  header: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  headerText: { ...type.small, color: colors.textTertiary },
  list: {
    gap: spacing.sm + 2,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  item: { flexDirection: "row", alignItems: "flex-start", gap: spacing.sm },
  itemText: { flex: 1, minWidth: 0 },
  question: { ...type.small, color: colors.textTertiary },
  answer: { ...type.small, color: colors.text },
}));
