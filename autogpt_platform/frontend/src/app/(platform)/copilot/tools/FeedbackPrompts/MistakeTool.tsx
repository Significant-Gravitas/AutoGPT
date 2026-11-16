"use client";

import { OttoMistakeCard } from "@/components/organisms/FeedbackPrompts/OttoMistakeCard";
import type { ToolUIPart } from "ai";
import { type FeedbackCardTarget, readFeedbackPrompt } from "./helpers";
import { useFeedbackCard } from "./useFeedbackCard";

interface Props {
  part: ToolUIPart;
  target: FeedbackCardTarget;
}

export function MistakeTool({ part, target }: Props) {
  const output = readFeedbackPrompt(part);
  if (!output) return null;
  return (
    <MistakePrompt
      key={output.promptId}
      promptId={output.promptId}
      target={target}
    />
  );
}

function MistakePrompt({
  promptId,
  target,
}: {
  promptId: string;
  target: FeedbackCardTarget;
}) {
  const card = useFeedbackCard({ kind: "otto_mistake", promptId, target });
  if (card.isLoading || !card.state || card.state.dismissed) return null;

  return (
    <OttoMistakeCard
      initialSent={card.state.answered}
      onReason={card.pickReason}
      onSubmit={card.submit}
      onDismiss={card.dismiss}
    />
  );
}
