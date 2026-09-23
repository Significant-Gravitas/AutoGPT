"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ElapsedTime } from "./components/ElapsedTime";
import { FailureState } from "./components/FailureState";
import { InsufficientState } from "./components/InsufficientState";
import { MicButton, OrbScreen } from "./components/MicButton";
import { PrivacyNote } from "./components/PrivacyNote";
import { RecordingControls } from "./components/RecordingControls/RecordingControls";
import { RecoveryPrompt } from "./components/RecoveryPrompt";
import { RestActions } from "./components/RestActions";
import { RevealGroup, RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { TypedFallback } from "./components/TypedFallback";
import { OrbControlButton } from "./components/OrbControlButton";
import { ScreenState, useBrainDumpStep } from "./useBrainDumpStep";

const FAILURE_HEADLINE = "That didn't go through.";
const TYPING_HEADLINE = "Write to me about your work";
// "Catch" fits a mishearing, not a typed answer — the typed reject asks
// for more instead of implying the system misheard.
const INSUFFICIENT_HEADLINES = {
  voice: "We didn't catch enough of that.",
  typed: "Tell us a bit more.",
} as const;
const TIME_LIMIT_CAPTION =
  "That's 30 minutes — the most we record in one go. Saving all of it…";

export function BrainDumpStep() {
  const dump = useBrainDumpStep();
  const isRecording = dump.screen === "recording";
  const isProcessing = dump.screen === "processing";
  const isMicScreen = dump.screen === "rest" || isRecording;
  const isTyping = dump.screen === "typing";
  const showPrivacyNote =
    dump.screen !== "failed" &&
    dump.screen !== "recovery" &&
    dump.screen !== "insufficient";
  // rest → recording → processing all share one orb, so it is never
  // unmounted between them: only the glyph and the ring change.
  const orbScreen = toOrbScreen(dump.screen);

  function orbClick(screen: OrbScreen) {
    if (screen === "processing") return undefined;
    if (screen === "failed") return dump.handleRetry;
    return screen === "rest" ? dump.handleStart : undefined;
  }

  return (
    <>
      <RevealGroup
        className={cn(
          "-mt-44 flex w-full flex-col items-center gap-8 px-4",
          isTyping ? "w-[calc(100vw-3rem)] max-w-3xl" : "max-w-2xl",
        )}
      >
        <div
          className={cn(
            "absolute right-4 top-4 flex items-center gap-2 sm:right-6 sm:top-6 sm:gap-5",
            isRecording && "hidden",
          )}
        >
          {/* Skipping mid-submit would advance the wizard a second time
              behind the finalize that is already in flight, landing past
              the last step on a blank screen. */}
          {!isProcessing && (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={dump.handleSkip}
              className="text-zinc-400 hover:text-zinc-700"
            >
              Skip for now
            </Button>
          )}
        </div>

        <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2 px-4 text-center">
          <RevealItem>
            <SwapFade
              swapKey={isRecording ? "timer" : "headline"}
              className="flex h-10 items-center justify-center"
            >
              {isRecording ? (
                <ElapsedTime seconds={dump.elapsedSeconds} />
              ) : (
                <Text variant="h4">
                  {stateHeadline(
                    dump.screen,
                    dump.headline,
                    dump.insufficientMode,
                  )}
                </Text>
              )}
            </SwapFade>
          </RevealItem>
        </div>

        {orbScreen && (
          <RevealItem
            blur={false}
            className="flex w-full flex-col items-center gap-4"
          >
            <MicButton screen={orbScreen} audioStream={dump.audioStream} />
            {/* One fixed-height slot holds whichever control the screen
                needs, so switching between the mic, the recording controls
                and the composer never moves the avatar or the headline. */}
            <div className="flex min-h-[236px] w-full flex-col items-center">
              {isTyping ? (
                <TypedFallback
                  value={dump.typedText}
                  onChange={dump.setTypedText}
                  onSubmit={dump.handleSubmitTyped}
                />
              ) : isRecording ? (
                <RecordingControls
                  onStop={dump.handleStop}
                  onSend={dump.handleDone}
                  onRetry={dump.handleRestart}
                  elapsedSeconds={dump.elapsedSeconds}
                  showSilenceNudge={dump.showSilenceNudge}
                  isOffline={dump.isOffline}
                />
              ) : orbScreen === "rest" ? (
                <RestActions
                  onTalk={dump.handleStart}
                  onWrite={dump.showTyping}
                />
              ) : (
                orbScreen === "failed" && (
                  <OrbControlButton
                    screen={orbScreen}
                    onClick={orbClick(orbScreen)}
                  />
                )
              )}
            </div>
            {/* Both slots keep their height across rest → recording →
                processing, so advancing a screen swaps their contents without
                nudging the orb or the headline. Failure has its own layout
                below the orb and needs neither. */}
            {orbScreen !== "failed" && !isRecording && (
              <>
                <div className="flex h-10 w-full items-center justify-center">
                  <SwapFade
                    swapKey={orbScreen}
                    className="flex w-full justify-center"
                  >
                    <OrbCaption
                      screen={orbScreen}
                      reachedTimeLimit={dump.reachedTimeLimit}
                    />
                  </SwapFade>
                </div>
                <div className="flex h-10 items-center justify-center" />
              </>
            )}
          </RevealItem>
        )}

        {dump.screen === "recovery" && dump.recoverable && (
          <RevealItem>
            <RecoveryPrompt
              durationSecs={dump.recoverable.durationSecs}
              onResume={dump.handleResumeRecovered}
              onDiscard={dump.handleDiscardRecovered}
            />
          </RevealItem>
        )}

        {dump.screen === "failed" && (
          <RevealItem>
            <FailureState
              onDownload={dump.handleDownloadRecording}
              onSkip={dump.handleSkip}
            />
          </RevealItem>
        )}

        {dump.screen === "insufficient" && (
          <RevealItem>
            <InsufficientState
              mode={dump.insufficientMode}
              canRecord={!dump.isMicBlocked}
              onRecordAgain={dump.handleRestart}
              onTypeInstead={dump.showTyping}
              onSkip={dump.handleSkip}
            />
          </RevealItem>
        )}
      </RevealGroup>

      {/* Viewport-anchored, and kept outside the reveal group: an ancestor
          that animates `filter` or `transform` would turn these into
          absolutely positioned elements. */}
      {!isRecording &&
        (isMicScreen ||
          isTyping ||
          dump.screen === "failed" ||
          dump.screen === "recovery") && (
          <div className="fixed inset-x-0 bottom-32 flex justify-center px-4">
            <SwapFade swapKey={dump.screen}>
              {dump.screen === "failed" && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={dump.showTyping}
                  className="underline underline-offset-4"
                >
                  type instead
                </Button>
              )}
              {isTyping && !dump.isMicBlocked && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={dump.showRecording}
                  className="underline underline-offset-4"
                >
                  record instead
                </Button>
              )}
              {dump.screen === "recovery" && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={dump.handleTypeInsteadOfRecovered}
                  className="underline underline-offset-4"
                >
                  type instead
                </Button>
              )}
            </SwapFade>
          </div>
        )}

      {showPrivacyNote && !isRecording && <PrivacyNote />}
    </>
  );
}

function toOrbScreen(screen: ScreenState): OrbScreen | null {
  // "insufficient" gets no orb: the orb's failed state retries the same
  // take, and re-submitting a rejected take can only be rejected again.
  // Typing keeps the resting avatar and swaps only the control under it.
  if (screen === "recovery" || screen === "insufficient") return null;
  if (screen === "typing") return "rest";
  return screen;
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

function OrbCaption({
  screen,
  reachedTimeLimit,
}: {
  screen: OrbScreen;
  reachedTimeLimit: boolean;
}) {
  if (screen === "processing") {
    return (
      <Text variant="body" tone="muted">
        {reachedTimeLimit ? TIME_LIMIT_CAPTION : "Got it. One second…"}
      </Text>
    );
  }

  return null;
}
