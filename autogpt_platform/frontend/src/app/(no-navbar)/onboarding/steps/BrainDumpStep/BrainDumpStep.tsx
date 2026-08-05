"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ElapsedTime } from "./components/ElapsedTime";
import { FailureState } from "./components/FailureState";
import { DEFAULT_GLASS_PARAMS } from "@/components/molecules/GlassOrb/GlassSurface";
import { MicButton, OrbScreen } from "./components/MicButton";
import { PrivacyNote } from "./components/PrivacyNote";
import { RecordingStatus } from "./components/RecordingStatus";
import { RecoveryPrompt } from "./components/RecoveryPrompt";
import { RevealGroup, RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import { TapHint } from "./components/TapHint";
import { TypedFallback } from "./components/TypedFallback";
import { ringProgress } from "./helpers";
import { ScreenState, useBrainDumpStep } from "./useBrainDumpStep";

const MIC_CAPTION = "Tap and talk. Most people go 2 to 3 minutes.";
const FAILURE_HEADLINE = "That didn't go through.";
const TIME_LIMIT_CAPTION =
  "That's 30 minutes — the most we record in one go. Saving all of it…";

export function BrainDumpStep() {
  const dump = useBrainDumpStep();
  const isRecording = dump.screen === "recording";
  const isProcessing = dump.screen === "processing";
  const isMicScreen = dump.screen === "rest" || isRecording;
  const isTyping = dump.screen === "typing";
  const showSubline = dump.screen !== "failed" && dump.screen !== "recovery";
  // rest → recording → processing all share one orb, so it is never
  // unmounted between them: only the glyph and the ring change.
  const orbScreen = toOrbScreen(dump.screen);

  function orbClick(screen: OrbScreen) {
    if (screen === "processing") return undefined;
    if (screen === "failed") return dump.handleRetry;
    return screen === "recording" ? dump.handleDone : dump.handleStart;
  }

  return (
    <>
      <RevealGroup
        className={cn(
          "-mt-44 flex w-full flex-col items-center gap-12 px-4",
          // The composer needs more room than the orb screens do.
          isTyping ? "max-w-4xl" : "max-w-2xl",
        )}
      >
        <div className="absolute right-6 top-6 flex items-center gap-5">
          {isRecording && (
            <Button
              variant="secondary"
              size="small"
              onClick={dump.handleRestart}
            >
              Restart
            </Button>
          )}
          {/* Skipping mid-submit would advance the wizard a second time
              behind the finalize that is already in flight, landing past
              the last step on a blank screen. */}
          {!isProcessing && (
            <button
              type="button"
              onClick={dump.handleSkip}
              className="text-sm text-zinc-400 transition-colors hover:text-zinc-700"
            >
              Skip for now
            </button>
          )}
        </div>

        <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2 px-4 text-center">
          <RevealItem>
            <Text variant="h3">
              {dump.screen === "failed" ? FAILURE_HEADLINE : dump.headline}
            </Text>
          </RevealItem>
          {showSubline && (
            <RevealItem>
              <Text
                variant="lead"
                className="!text-zinc-500 md:whitespace-nowrap"
              >
                Just talk.{" "}
                <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
                  AutoPilot
                </span>{" "}
                listens, remembers, and starts taking work off your plate.
              </Text>
            </RevealItem>
          )}
        </div>

        {orbScreen && (
          <RevealItem className="flex flex-col items-center gap-4">
            <MicButton
              screen={orbScreen}
              progress={ringProgress(dump.elapsedSeconds)}
              audioStream={dump.audioStream}
              glassParams={DEFAULT_GLASS_PARAMS}
              onClick={orbClick(orbScreen)}
            />
            {/* Both slots keep their height across rest → recording →
                processing, so advancing a screen swaps their contents without
                nudging the orb or the headline. Failure has its own layout
                below the orb and needs neither. */}
            {orbScreen !== "failed" && (
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
                <div className="flex h-10 items-center justify-center">
                  {isRecording && <ElapsedTime seconds={dump.elapsedSeconds} />}
                </div>
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

        {dump.screen === "typing" && (
          <RevealItem className="w-full">
            <TypedFallback
              value={dump.typedText}
              onChange={dump.setTypedText}
              onSubmit={dump.handleSubmitTyped}
            />
          </RevealItem>
        )}
      </RevealGroup>

      {/* Viewport-anchored, and kept outside the reveal group: an ancestor
          that animates `filter` or `transform` would turn these into
          absolutely positioned elements. */}
      {(isMicScreen ||
        isTyping ||
        dump.screen === "failed" ||
        dump.screen === "recovery") && (
        <div className="fixed inset-x-0 bottom-32 flex justify-center px-4">
          <SwapFade swapKey={dump.screen}>
            {isRecording && (
              <div className="flex flex-col items-center gap-3">
                <RecordingStatus
                  elapsedSeconds={dump.elapsedSeconds}
                  showSilenceNudge={dump.showSilenceNudge}
                  isOffline={dump.isOffline}
                  isSavedLocally={dump.isSavedLocally}
                />
                <Button
                  variant="primary"
                  size="large"
                  onClick={dump.handleDone}
                >
                  I&apos;m done
                </Button>
              </div>
            )}
            {(dump.screen === "rest" || dump.screen === "failed") && (
              <Button
                variant="ghost"
                size="small"
                onClick={dump.showTyping}
                className="underline underline-offset-4"
              >
                type instead
              </Button>
            )}
            {isTyping && !dump.isMicBlocked && (
              <Button
                variant="ghost"
                size="small"
                onClick={dump.showRecording}
                className="underline underline-offset-4"
              >
                record instead
              </Button>
            )}
            {dump.screen === "recovery" && (
              <Button
                variant="ghost"
                size="small"
                onClick={dump.handleTypeInsteadOfRecovered}
                className="underline underline-offset-4"
              >
                type instead
              </Button>
            )}
          </SwapFade>
        </div>
      )}

      {showSubline && <PrivacyNote />}
    </>
  );
}

function toOrbScreen(screen: ScreenState): OrbScreen | null {
  if (screen === "typing" || screen === "recovery") return null;
  return screen;
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
      <Text variant="lead" className="!text-base !text-zinc-500">
        {reachedTimeLimit ? TIME_LIMIT_CAPTION : "Got it. One second…"}
      </Text>
    );
  }

  // The failure copy sits with its buttons in FailureState, so it is not
  // held back by the swap's exit animation.
  if (screen === "failed") return null;

  if (screen === "recording") return null;

  return <TapHint caption={MIC_CAPTION} />;
}
