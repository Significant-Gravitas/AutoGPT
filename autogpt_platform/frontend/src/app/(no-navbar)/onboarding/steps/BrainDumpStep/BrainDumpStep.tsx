"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AmbientPrompts } from "./components/AmbientPrompts";
import { FailureState } from "./components/FailureState";
import { IdeasPauseToggle } from "./components/IdeasPauseToggle";
import { InsufficientState } from "./components/InsufficientState";
import { MicButton, OrbScreen } from "./components/MicButton";
import { ModeSwitchLink } from "./components/ModeSwitchLink";
import { OttoMightAsk } from "./components/OttoMightAsk";
import { PrivacyNote } from "./components/PrivacyNote";
import { RecordingControls } from "./components/RecordingControls/RecordingControls";
import { RecoveryPrompt } from "./components/RecoveryPrompt";
import { RestActions } from "./components/RestActions";
import { RevealGroup, RevealItem } from "@/components/atoms/Reveal/Reveal";
import { SkipButton } from "./components/SkipButton";
import { StepHeader } from "./components/StepHeader";
import { TypingComposer } from "./components/TypingComposer";
import { OrbControlButton } from "./components/OrbControlButton";
import { toOrbScreen } from "./helpers";
import { useBrainDumpStep } from "./useBrainDumpStep";

const MIC_UNAVAILABLE_COPY =
  "We couldn't open your microphone. You can write it instead.";
const TIME_LIMIT_CAPTION =
  "That's 30 minutes, the most we record in one go. Saving all of it…";

export function BrainDumpStep() {
  const dump = useBrainDumpStep();
  const isRecording = dump.screen === "recording";
  const isProcessing = dump.screen === "processing";
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

  function showAnotherQuestion() {
    dump.prompt.advance("click");
  }

  return (
    <>
      <SkipButton
        isHidden={isRecording}
        isProcessing={isProcessing}
        onSkip={dump.handleSkip}
      />

      {/* Wide enough that the starter ideas clear the typing composer. */}
      <div className="relative flex w-full max-w-screen-2xl flex-col items-center justify-center self-stretch">
        <IdeasPauseToggle
          screen={dump.screen}
          isPaused={dump.prompt.isPaused}
          onToggle={dump.prompt.togglePause}
        />
        <AmbientPrompts
          role={dump.role}
          screen={dump.screen}
          isPaused={dump.prompt.isPaused}
          onTogglePause={dump.prompt.togglePause}
        />
        <RevealGroup
          className={cn(
            "relative z-10 flex w-full flex-col items-center gap-6 px-4",
            isTyping
              ? "w-[calc(100vw-3rem)] max-w-3xl [@media(max-height:800px)]:gap-4"
              : "max-w-2xl",
          )}
        >
          <StepHeader
            screen={dump.screen}
            restHeadline={dump.headline}
            insufficientMode={dump.insufficientMode}
            elapsedSeconds={dump.elapsedSeconds}
            question={dump.prompt.question.text}
            onTogglePause={dump.prompt.togglePause}
          />

          {orbScreen && (
            <RevealItem
              blur={false}
              className={cn(
                "flex w-full flex-col items-center gap-4",
                isTyping && "[@media(max-height:800px)]:gap-2",
              )}
            >
              <MicButton screen={orbScreen} audioStream={dump.audioStream} />
              <div className="flex w-full flex-col items-center">
                {isTyping ? (
                  <TypingComposer
                    value={dump.typedText}
                    onChange={dump.setTypedText}
                    isTooLong={dump.isTypedTooLong}
                    onSubmit={dump.handleSubmitTyped}
                    question={dump.prompt.question.text}
                    announcement={dump.prompt.announcement}
                    onAnotherQuestion={showAnotherQuestion}
                    onTogglePause={dump.prompt.togglePause}
                    isMicBlocked={dump.isMicBlocked}
                    isMicUnavailable={dump.isMicUnavailable}
                    micPermission={dump.micPermission}
                    onRetryMicrophone={dump.handleRetryMicrophone}
                  />
                ) : isRecording ? (
                  <>
                    <OttoMightAsk
                      question={dump.prompt.question.text}
                      onAnother={showAnotherQuestion}
                      announcement={dump.prompt.announcement}
                      onTogglePause={dump.prompt.togglePause}
                    />
                    <RecordingControls
                      onStop={dump.handleStop}
                      onSend={dump.handleDone}
                      onRetry={dump.handleRestart}
                      elapsedSeconds={dump.elapsedSeconds}
                      showSilenceNudge={dump.showSilenceNudge}
                      isOffline={dump.isOffline}
                    />
                  </>
                ) : orbScreen === "rest" ? (
                  <>
                    <RestActions
                      onTalk={dump.handleStart}
                      onWrite={dump.showTyping}
                      writeFirst={dump.isMicUnavailable}
                    />
                    {dump.isMicUnavailable && (
                      <Text
                        variant="body"
                        tone="muted"
                        role="alert"
                        className="mt-3 text-center"
                      >
                        {MIC_UNAVAILABLE_COPY}
                      </Text>
                    )}
                  </>
                ) : (
                  orbScreen === "failed" && (
                    <OrbControlButton
                      screen={orbScreen}
                      onClick={orbClick(orbScreen)}
                    />
                  )
                )}
              </div>
              {isProcessing && (
                <Text variant="body" tone="muted" className="text-center">
                  {dump.reachedTimeLimit
                    ? TIME_LIMIT_CAPTION
                    : "Got it. One second…"}
                </Text>
              )}
            </RevealItem>
          )}

          {dump.screen === "recovery" && dump.recoverable && (
            <RevealItem>
              <RecoveryPrompt
                durationSecs={dump.recoverable.durationSecs}
                onResume={dump.handleResumeRecovered}
                onDownload={dump.handleDownloadRecording}
                onDiscard={dump.handleDiscardRecovered}
              />
            </RevealItem>
          )}

          {dump.screen === "failed" && (
            <RevealItem>
              <FailureState
                mode={dump.submittedMode}
                reason={dump.failureReason}
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

          <ModeSwitchLink
            screen={dump.screen}
            submittedMode={dump.submittedMode}
            onTypeInstead={dump.showTyping}
            onRecordInstead={dump.handleRecordInstead}
            onTypeInsteadOfRecovered={dump.handleTypeInsteadOfRecovered}
          />

          {showPrivacyNote && !isRecording && <PrivacyNote />}
        </RevealGroup>
      </div>
    </>
  );
}
