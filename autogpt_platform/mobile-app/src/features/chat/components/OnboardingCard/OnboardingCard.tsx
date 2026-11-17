import { ArrowLeft01Icon, SentIcon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { createStyles, radius, spacing, type } from "@/lib/theme";

import type { StepPart } from "../../steps";
import { AnsweredOnboarding } from "./components/AnsweredOnboarding";
import { OnboardingQuestion } from "./components/OnboardingQuestion";
import { parseOnboarding, parseSentAnswers, type OnboardingStep } from "./helpers";
import { useOnboardingCard } from "./useOnboardingCard";

interface Props {
  part: StepPart;
  /** Text of the message after the card, or null while the card is the last thing in the chat. */
  replyText: string | null;
  isBusy: boolean;
  onSend: (text: string) => boolean;
}

/** The questions a newly hired expert opens its chat with, one at a time. */
export function OnboardingCard({ part, replyText, isBusy, onSend }: Props) {
  const styles = useStyles();
  const steps =
    part.state === "output-available" ? parseOnboarding(part.output) : null;

  if (!steps) {
    const isSettled =
      part.state === "output-available" || part.state === "output-error";
    return (
      <Text style={styles.status}>
        {isSettled ? "Couldn't open the setup questions." : "Getting set up…"}
      </Text>
    );
  }

  // Any reply settles the card, as on the web: the answers, a skip or anything else.
  if (replyText !== null) {
    return <AnsweredOnboarding answers={parseSentAnswers(replyText, steps)} />;
  }

  return <OnboardingForm steps={steps} isBusy={isBusy} onSend={onSend} />;
}

interface FormProps {
  steps: OnboardingStep[];
  isBusy: boolean;
  onSend: (text: string) => boolean;
}

function OnboardingForm({ steps, isBusy, onSend }: FormProps) {
  const styles = useStyles();
  const card = useOnboardingCard({ steps, onSend });

  if (card.sentText !== null) {
    return <AnsweredOnboarding answers={parseSentAnswers(card.sentText, steps)} />;
  }

  const total = steps.length;

  return (
    <View style={styles.card}>
      <View
        accessibilityRole="progressbar"
        accessibilityLabel="Setup progress"
        accessibilityValue={{ min: 0, max: total, now: card.answeredCount }}
        style={styles.track}
      >
        <View
          style={[styles.fill, { width: `${(card.answeredCount / total) * 100}%` }]}
        />
      </View>
      <View style={styles.body}>
        <OnboardingQuestion
          key={card.step.keyword}
          number={card.current + 1}
          step={card.step}
          answer={card.answer}
          onChoose={card.choose}
          onChangeText={card.changeText}
        />
      </View>
      <View style={styles.footer}>
        {card.current > 0 ? (
          <IconButton
            icon={ArrowLeft01Icon}
            label="Previous question"
            onPress={card.back}
          />
        ) : (
          <View />
        )}
        <View style={styles.actions}>
          <Button
            label="Skip questions"
            variant="secondary"
            size="small"
            disabled={isBusy}
            onPress={card.skip}
          />
          <Button
            label={card.isLast ? "Send" : "Next"}
            accessibilityLabel={card.isLast ? "Send answers" : "Next question"}
            leadingIcon={card.isLast ? SentIcon : undefined}
            size="small"
            disabled={!card.isReady || isBusy}
            onPress={card.next}
          />
        </View>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  status: { ...type.small, color: colors.textTertiary },
  card: {
    overflow: "hidden",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.surface,
  },
  track: { height: 4, backgroundColor: colors.surfaceRaised },
  fill: { height: "100%", backgroundColor: colors.action },
  body: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg },
  footer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.sm,
    padding: spacing.lg,
  },
  actions: { flexDirection: "row", gap: spacing.sm },
}));
