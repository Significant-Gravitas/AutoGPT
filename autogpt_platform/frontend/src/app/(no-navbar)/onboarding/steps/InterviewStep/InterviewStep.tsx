"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { MicBlockedNotice } from "../BrainDumpStep/components/MicBlockedNotice";
import { SkipButton } from "../BrainDumpStep/components/SkipButton";
import { InterviewChat } from "./components/InterviewChat";
import { EndingPicker } from "./components/Endings/components/EndingPicker";
import { LiveEnding } from "./components/Endings/components/LiveEnding";
import { StartOver } from "./components/Endings/components/StartOver";
import { VoiceCall } from "./components/VoiceCall/VoiceCall";
import { startFailureCopy } from "./helpers";
import { useInterviewStep } from "./useInterviewStep";

export function InterviewStep() {
  const interview = useInterviewStep();

  return (
    <>
      {interview.canSkip && (
        <SkipButton
          isHidden={false}
          isProcessing={interview.isLeaving}
          onSkip={interview.skip}
        />
      )}

      <div className="flex w-full max-w-2xl flex-col items-center gap-6 px-4">
        <div className="flex flex-col items-center gap-2 text-center">
          <Text variant="h3">Tell Otto about your work</Text>
          <Text variant="body" tone="muted">
            A few quick questions, so your team starts on what matters to you.
          </Text>
        </div>

        {interview.mode === "choose" || interview.mode === "talk" ? (
          <VoiceCall call={interview.voice} />
        ) : (
          <>
            {interview.isEndingPickerOn && !interview.isOver && (
              <EndingPicker mode="text" onPick={interview.repickEnding} />
            )}
            {!interview.isOver && interview.sessionId && (
              <LiveEnding
                arm={interview.liveEnding.arm}
                details={interview.liveEnding.details}
                canFinish={interview.canFinishEarly}
                isLeaving={interview.isLeaving}
                onFinish={() => void interview.finish()}
              />
            )}
            <MicBlockedNotice
              isBlocked={interview.isMicBlocked}
              isUnavailable={interview.isMicUnavailable}
              permission={interview.micPermission}
              onRetry={() => void interview.retryMicrophone()}
            />
            {interview.startFailure ? (
              <div
                role="alert"
                className="flex flex-col items-center gap-3 text-center"
              >
                <Text variant="body">
                  {startFailureCopy(interview.startFailure)}
                </Text>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={() => void interview.retryStart()}
                >
                  Try again
                </Button>
              </div>
            ) : interview.sessionId ? (
              <InterviewChat
                sessionId={interview.sessionId}
                messages={interview.messages}
                status={interview.status}
                error={interview.error}
                queuedMessages={interview.queuedMessages}
                isHydrating={interview.isHydrating}
                isTurnSettling={interview.isTurnSettling}
                isTurnRunning={interview.isTurnRunning}
                isOver={interview.isOver}
                canFinishEarly={interview.canFinishEarly}
                isLeaving={interview.isLeaving}
                finishFailure={interview.finishFailure}
                ending={interview.typedEnding}
                startOver={
                  <StartOver
                    canCall={interview.typedStartOver.canCall}
                    hasKept={interview.typedStartOver.hasKept}
                    isDisabled={interview.isLeaving}
                    onOpen={interview.typedStartOver.onOpen}
                    onCallAgain={interview.typedStartOver.onCallAgain}
                    onTypeInstead={interview.typedStartOver.onTypeInstead}
                  />
                }
                onSend={interview.onSend}
                onStop={interview.stop}
                onFinish={interview.finish}
              />
            ) : (
              <Text variant="body" tone="muted" role="status">
                Starting your interview…
              </Text>
            )}
          </>
        )}
      </div>
    </>
  );
}
