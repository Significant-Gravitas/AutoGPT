import { ArrowRight01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Pressable, Text, TextInput, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import {
  optionLetter,
  type OnboardingStep,
  type StepAnswer,
} from "../helpers";

interface Props {
  number: number;
  step: OnboardingStep;
  answer: StepAnswer | undefined;
  onChoose: (option: string) => void;
  onChangeText: (text: string) => void;
}

/** One question's lettered options and "Other". Keyed per step, so the text field starts fresh. */
export function OnboardingQuestion({
  number,
  step,
  answer,
  onChoose,
  onChangeText,
}: Props) {
  const styles = useStyles();
  const colors = useColors();
  const custom = answer?.custom ?? "";
  const [isTyping, setIsTyping] = useState(
    custom.trim().length > 0 || step.options.length === 0,
  );

  function handleChoose(option: string) {
    if (!step.allowMultiple) setIsTyping(false);
    onChoose(option);
  }

  function handleOther() {
    if (!step.allowMultiple) onChangeText(custom);
    setIsTyping(true);
  }

  return (
    <View style={styles.question}>
      <View style={styles.heading}>
        <View style={styles.number}>
          <Text style={styles.numberText}>{number}</Text>
          <Icon icon={ArrowRight01Icon} size={12} color={colors.textTertiary} />
        </View>
        <Text style={styles.title}>{step.question}</Text>
      </View>
      {step.allowMultiple ? (
        <Text style={styles.hint}>Choose as many as you like</Text>
      ) : null}
      {step.options.map((option, index) => {
        const isSelected = answer?.picks.includes(option) ?? false;
        return (
          <Pressable
            key={option}
            accessibilityRole={step.allowMultiple ? "checkbox" : "radio"}
            accessibilityState={{ checked: isSelected }}
            onPress={() => handleChoose(option)}
            style={({ pressed }) => [
              styles.option,
              isSelected && styles.optionSelected,
              pressed && styles.pressed,
            ]}
          >
            <Badge letter={optionLetter(index)} isSelected={isSelected} />
            <Text style={[styles.optionText, isSelected && styles.selectedText]}>
              {option}
            </Text>
            {isSelected ? (
              <Icon icon={Tick02Icon} size={16} color={colors.onAction} />
            ) : null}
          </Pressable>
        );
      })}
      {isTyping ? (
        <TextInput
          value={custom}
          onChangeText={onChangeText}
          placeholder={step.allowMultiple ? "Add your own answer" : "Type your answer"}
          placeholderTextColor={colors.textTertiary}
          accessibilityLabel={step.question}
          autoFocus={step.options.length > 0}
          multiline
          style={styles.input}
        />
      ) : (
        <Pressable
          accessibilityRole="button"
          onPress={handleOther}
          style={({ pressed }) => [styles.other, pressed && styles.pressed]}
        >
          <Badge letter={optionLetter(step.options.length)} isSelected={false} />
          <Text style={styles.otherText}>Other</Text>
        </Pressable>
      )}
    </View>
  );
}

interface BadgeProps {
  letter: string;
  isSelected: boolean;
}

function Badge({ letter, isSelected }: BadgeProps) {
  const styles = useStyles();
  return (
    <View
      importantForAccessibility="no-hide-descendants"
      accessibilityElementsHidden
      style={[styles.badge, isSelected && styles.badgeSelected]}
    >
      <Text style={[styles.badgeText, isSelected && styles.selectedText]}>
        {letter}
      </Text>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  question: { gap: spacing.sm },
  heading: { flexDirection: "row", gap: spacing.sm, marginBottom: spacing.xs },
  number: { flexDirection: "row", alignItems: "center", gap: 2, height: 24 },
  numberText: { ...type.small, color: colors.textTertiary },
  title: { ...type.heading, color: colors.text, flex: 1 },
  hint: { ...type.small, color: colors.textSecondary, marginTop: -spacing.xs },
  option: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 48,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
  },
  optionSelected: { backgroundColor: colors.action, borderColor: colors.action },
  optionText: { ...type.body, color: colors.text, flex: 1 },
  selectedText: { color: colors.onAction },
  pressed: { opacity: 0.7 },
  badge: {
    width: 24,
    height: 24,
    borderRadius: 6,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.surface,
  },
  badgeSelected: { borderColor: colors.onAction, backgroundColor: "transparent" },
  badgeText: { fontSize: 12, fontWeight: "500", lineHeight: 16, color: colors.textSecondary },
  other: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 48,
    paddingHorizontal: spacing.md,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderStyle: "dashed",
    borderColor: colors.borderBase,
  },
  otherText: { ...type.body, color: colors.textSecondary },
  input: {
    ...type.body,
    color: colors.text,
    minHeight: 72,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm + 2,
    paddingBottom: spacing.sm + 2,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
    textAlignVertical: "top",
  },
}));
