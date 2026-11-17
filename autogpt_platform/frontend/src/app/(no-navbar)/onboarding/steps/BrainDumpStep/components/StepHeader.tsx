"use client";

import { RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { Text } from "@/components/atoms/Text/Text";
import type { ScreenState } from "../useBrainDumpStep";
import { ElapsedTime } from "./ElapsedTime";
import {
  FEW_SENTENCES_HINT,
  OFF_YOUR_PLATE_QUESTION,
} from "./insufficient-copy";
import { OttoMightAsk } from "./OttoMightAsk";
import { OttoQuestionCard } from "./OttoQuestionCard";

const FAILURE_HEADLINE = "That didn't go through.";
const TYPING_HEADLINE = "Write to me about your work";

interface Props {
  screen: ScreenState;
  restHeadline: string;
  // After the not-enough screen, Otto's question stays the headline.
  isFollowUp: boolean;
  elapsedSeconds: number;
  question: string;
  onTogglePause: () => void;
}

export function StepHeader({
  screen,
  restHeadline,
  isFollowUp,
  elapsedSeconds,
  question,
  onTogglePause,
}: Props) {
  // Otto's big question is the heading of the not-enough screen.
  if (screen === "insufficient") return null;
  const isRecording = screen === "recording";

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2 px-4 text-center">
      {isRecording && isFollowUp && (
        <RevealItem className="flex w-full justify-center">
          <OttoQuestionCard />
        </RevealItem>
      )}
      <RevealItem>
        <SwapFade
          swapKey={isRecording ? "timer" : "headline"}
          className="flex min-h-10 items-center justify-center"
        >
          {isRecording ? (
            <ElapsedTime seconds={elapsedSeconds} />
          ) : (
            <Text variant="h3">
              {stateHeadline(screen, restHeadline, isFollowUp)}
            </Text>
          )}
        </SwapFade>
      </RevealItem>
      {screen === "rest" && (
        <RevealItem>
          <div className="flex flex-col items-center gap-4">
            <p className="max-w-xl font-sans text-body-lg font-normal text-foreground-tertiary md:text-lead">
              Tell me what you do and what eats up your week.{" "}
              <Text variant="body-medium" as="span">
                Otto uses it to pick your experts and plan your first week.
              </Text>{" "}
              A minute is plenty, and you can type instead.
            </p>
            {/* On wide screens the ambient cards carry these questions, and
              they are hidden from assistive tech, so this one stays in the
              accessibility tree. */}
            <OttoMightAsk
              question={question}
              compact
              onTogglePause={onTogglePause}
              className="xl:sr-only"
            />
          </div>
        </RevealItem>
      )}
      {screen === "typing" && (
        <RevealItem>
          {isFollowUp ? (
            <Text variant="body" tone="muted" className="max-w-2xl">
              {FEW_SENTENCES_HINT}
            </Text>
          ) : (
            <Text variant="body" tone="muted" className="max-w-2xl">
              What repeats every week, and what would you hand off first?{" "}
              <Text variant="body-medium" as="span">
                {FEW_SENTENCES_HINT}
              </Text>
            </Text>
          )}
        </RevealItem>
      )}
    </div>
  );
}

function stateHeadline(
  screen: ScreenState,
  restHeadline: string,
  isFollowUp: boolean,
) {
  if (screen === "failed") return FAILURE_HEADLINE;
  if (screen === "typing")
    return isFollowUp ? OFF_YOUR_PLATE_QUESTION : TYPING_HEADLINE;
  return restHeadline;
}
