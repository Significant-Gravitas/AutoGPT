"use client";

import { RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { Text } from "@/components/atoms/Text/Text";
import type { ScreenState } from "../useBrainDumpStep";
import { ElapsedTime } from "./ElapsedTime";
import { OttoMightAsk } from "./OttoMightAsk";

const FAILURE_HEADLINE = "That didn't go through.";
const TYPING_HEADLINE = "Write to me about your work";
// "Catch" fits a mishearing, not a typed answer — the typed reject asks
// for more instead of implying the system misheard.
const INSUFFICIENT_HEADLINES = {
  voice: "We didn't catch enough of that.",
  typed: "Tell us a bit more.",
} as const;

interface Props {
  screen: ScreenState;
  restHeadline: string;
  insufficientMode: "voice" | "typed";
  elapsedSeconds: number;
  question: string;
  onTogglePause: () => void;
}

export function StepHeader({
  screen,
  restHeadline,
  insufficientMode,
  elapsedSeconds,
  question,
  onTogglePause,
}: Props) {
  const isRecording = screen === "recording";

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2 px-4 text-center">
      <RevealItem>
        <SwapFade
          swapKey={isRecording ? "timer" : "headline"}
          className="flex min-h-10 items-center justify-center"
        >
          {isRecording ? (
            <ElapsedTime seconds={elapsedSeconds} />
          ) : (
            <Text variant="h3">
              {stateHeadline(screen, restHeadline, insufficientMode)}
            </Text>
          )}
        </SwapFade>
      </RevealItem>
      {screen === "rest" && (
        <RevealItem className="flex flex-col items-center gap-4">
          <Text variant="large" tone="muted" className="max-w-xl md:text-lead">
            Tell me what you do and what eats up your week.{" "}
            <Text variant="body-medium" as="span">
              Otto uses it to pick your experts and plan your first week.
            </Text>{" "}
            A minute is plenty, and you can type instead.
          </Text>
          {/* On wide screens the ambient cards carry these questions, and
              they are hidden from assistive tech, so this one stays in the
              accessibility tree. */}
          <OttoMightAsk
            question={question}
            compact
            onTogglePause={onTogglePause}
            className="xl:sr-only"
          />
        </RevealItem>
      )}
      {screen === "typing" && (
        <RevealItem>
          <Text variant="body" tone="muted" className="max-w-2xl">
            What repeats every week, and what would you hand off first?{" "}
            <Text variant="body-medium" as="span">
              A few sentences is plenty.
            </Text>
          </Text>
        </RevealItem>
      )}
    </div>
  );
}

function stateHeadline(
  screen: ScreenState,
  restHeadline: string,
  insufficientMode: "voice" | "typed",
) {
  if (screen === "failed") return FAILURE_HEADLINE;
  if (screen === "typing") return TYPING_HEADLINE;
  if (screen === "insufficient")
    return INSUFFICIENT_HEADLINES[insufficientMode];
  return restHeadline;
}
