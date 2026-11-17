"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { finishFailureCopy, type StartFailure } from "../helpers";
import { type EndingData, InterviewEnding } from "./Endings/InterviewEnding";
import type { EndingArm } from "./Endings/helpers";

type ChatProps = React.ComponentProps<typeof ChatMessagesContainer>;

interface Props {
  sessionId: string;
  messages: ChatProps["messages"];
  status: ChatProps["status"];
  error: ChatProps["error"];
  queuedMessages: ChatProps["queuedMessages"];
  isHydrating: boolean;
  isTurnSettling: boolean;
  isTurnRunning: boolean;
  isOver: boolean;
  canFinishEarly: boolean;
  isLeaving: boolean;
  finishFailure: StartFailure | null;
  ending: {
    arm: EndingArm | null;
    data: EndingData;
    isTesterPick: boolean;
    isSettled: boolean;
  };
  // The way to do the interview over, under its ending.
  startOver: ReactNode;
  onSend: (message: string) => Promise<void>;
  onStop: () => void;
  onFinish: (corrections?: string | null) => Promise<void>;
}

export function InterviewChat({
  sessionId,
  messages,
  status,
  error,
  queuedMessages,
  isHydrating,
  isTurnSettling,
  isTurnRunning,
  isOver,
  canFinishEarly,
  isLeaving,
  finishFailure,
  ending,
  startOver,
  onSend,
  onStop,
  onFinish,
}: Props) {
  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <div
        role="region"
        aria-label="Interview with Otto"
        className="flex h-[60vh] max-h-[40rem] min-h-[20rem] w-full flex-col overflow-hidden rounded-xl border border-border-base bg-surface"
      >
        <div className="flex min-h-0 flex-1 flex-col">
          <ChatMessagesContainer
            messages={messages}
            status={status}
            error={error}
            isLoading={isHydrating}
            sessionID={sessionId}
            queuedMessages={queuedMessages}
            isTurnSettling={isTurnSettling}
            variant="compact"
            // The step's heading names Otto; a chip floating over the top
            // of a card this small would cover the transcript's first line.
            showThreadHeader={false}
          />
        </div>
        <div className="flex shrink-0 flex-col gap-2 border-t border-separator px-3 py-3">
          {isOver ? (
            <div className="flex justify-center">
              <InterviewEnding
                arm={ending.arm}
                data={ending.data}
                isLeaving={isLeaving}
                isTesterPick={ending.isTesterPick}
                isSettled={ending.isSettled}
                startOver={startOver}
                onConfirm={(corrections) => void onFinish(corrections)}
                fallback={
                  <Button
                    variant="primary"
                    size="large"
                    loading={isLeaving}
                    disabled={isLeaving}
                    onClick={() => void onFinish()}
                  >
                    Meet your team
                  </Button>
                }
              />
            </div>
          ) : (
            <div className="flex items-end gap-2">
              <div className="min-w-0 flex-1">
                <ChatInput
                  inputId="onboarding-interview-input"
                  onSend={onSend}
                  disabled={isLeaving}
                  isStreaming={isTurnRunning}
                  isTurnStarting={status === "submitted"}
                  onStop={onStop}
                  onEnqueue={onSend}
                  placeholder="Type your answer…"
                  hasSession={true}
                  variant="compact"
                  textOnly
                />
              </div>
              {canFinishEarly && (
                <Button
                  variant="secondary"
                  size="small"
                  loading={isLeaving}
                  disabled={isLeaving}
                  onClick={() => void onFinish()}
                >
                  I&apos;m done
                </Button>
              )}
            </div>
          )}
          {finishFailure && (
            <Text variant="small" tone="error" role="alert">
              {finishFailureCopy(finishFailure)}
            </Text>
          )}
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
