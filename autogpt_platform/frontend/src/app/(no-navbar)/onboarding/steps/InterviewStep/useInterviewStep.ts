import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import {
  finishOnboardingInterview,
  startOnboardingInterview,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  getV2GetSession,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { recordSkippedTake } from "../BrainDumpStep/useBrainDumpStep";
import { forgetShownPlan } from "../SubscriptionStep/components/PlanPanel/helpers";
import {
  hasAnswered,
  isEndedStage,
  isInterviewClosedError,
  STAGE_READ_INTERVAL_MS,
  STAGE_READ_WINDOW_MS,
  type StartFailure,
  startFailureOf,
  withoutKickoff,
} from "./helpers";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

export function useInterviewStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const setStepBusy = useOnboardingWizardStore((s) => s.setStepBusy);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [startFailure, setStartFailure] = useState<StartFailure | null>(null);
  const [startedStage, setStartedStage] = useState<string | null>(null);
  const [readStage, setReadStage] = useState<string | null>(null);
  const [isReadingStage, setIsReadingStage] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  const [hasFinishFailed, setHasFinishFailed] = useState(false);
  const isLeavingRef = useRef(false);
  const startedRef = useRef(false);
  const startAttemptRef = useRef(0);
  const turnRanRef = useRef(false);

  const sessionQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: !!sessionId,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnMount: true,
    },
  });
  const session =
    sessionQuery.data?.status === 200 ? sessionQuery.data.data : null;
  const hasActiveStream = !!session?.active_stream;

  const hydratedMessages = useMemo<UiMessages | undefined>(() => {
    if (!session || !sessionId) return undefined;
    return convertChatSessionMessagesToUiMessages(
      sessionId,
      session.messages ?? [],
      { isComplete: !hasActiveStream },
    ).messages as UiMessages;
  }, [session, sessionId, hasActiveStream]);

  const {
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error,
    isReconnecting,
    isFinishProbing,
  } = useCopilotStream({
    sessionId,
    hydratedMessages,
    hasActiveStream,
    refetchSession: sessionQuery.refetch,
    copilotModel: undefined,
  });

  const { queuedMessages, queueMessage } = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });

  const isTurnRunning = status === "streaming" || status === "submitted";
  const isOver =
    isEndedStage(startedStage) ||
    isEndedStage(session?.metadata?.onboarding_interview_stage) ||
    isEndedStage(readStage) ||
    isInterviewClosedError(error);

  useEffect(() => {
    setStepBusy(isLeaving);
    return () => setStepBusy(false);
  }, [isLeaving, setStepBusy]);

  // Otto's end signal closes the interview inside the turn; the reply cap and
  // the scope guard close it just after. So once a turn ends the stage is read
  // for a while, and the finish button waits until it has been.
  useEffect(() => {
    if (isTurnRunning) {
      turnRanRef.current = true;
      return;
    }
    if (!turnRanRef.current || !sessionId) return;
    turnRanRef.current = false;
    let cancelled = false;
    setIsReadingStage(true);
    async function readUntilOver(id: string) {
      const deadline = Date.now() + STAGE_READ_WINDOW_MS;
      while (!cancelled && Date.now() < deadline) {
        const stage = await fetchStage(id);
        if (cancelled) return;
        if (isEndedStage(stage)) {
          setReadStage(stage);
          break;
        }
        await pause(STAGE_READ_INTERVAL_MS);
      }
      if (!cancelled) setIsReadingStage(false);
    }
    void readUntilOver(sessionId);
    return () => {
      cancelled = true;
      setIsReadingStage(false);
    };
  }, [isTurnRunning, sessionId]);

  async function start() {
    const attempt = ++startAttemptRef.current;
    setStartFailure(null);
    try {
      const response = await startOnboardingInterview();
      if (attempt !== startAttemptRef.current) return;
      if (response.status !== 200) {
        setStartFailure({ kind: "failed" });
        return;
      }
      setStartedStage(response.data.stage ?? null);
      setSessionId(response.data.session_id);
    } catch (startError) {
      if (attempt !== startAttemptRef.current) return;
      const failure = startFailureOf(startError);
      if (failure === "over") {
        await finish();
        return;
      }
      setStartFailure(failure);
    }
  }

  async function finish() {
    if (isLeavingRef.current) return;
    isLeavingRef.current = true;
    setIsLeaving(true);
    setHasFinishFailed(false);
    try {
      await finishOnboardingInterview();
    } catch {
      isLeavingRef.current = false;
      setIsLeaving(false);
      setHasFinishFailed(true);
      return;
    }
    setIntroPath("A");
    forgetShownPlan(useOnboardingWizardStore.getState().userID);
    await saveWizardAnswers();
    nextStep();
  }

  async function skip() {
    if (isLeavingRef.current) return;
    isLeavingRef.current = true;
    setIsLeaving(true);
    setIntroPath("B");
    forgetShownPlan(useOnboardingWizardStore.getState().userID);
    await saveWizardAnswers();
    nextStep();
    await recordSkippedTake();
  }

  // Opened once per visit; a retry calls start itself.
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    void start();
  }, [start]);

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed || !sessionId || isOver) return;
    if (isTurnRunning) {
      try {
        await queueFollowUpMessage(sessionId, trimmed);
        queueMessage(trimmed);
      } catch (queueError) {
        // The turn ended while the message was on its way: send it instead.
        if (
          queueError instanceof Error &&
          queueError.name === "QueueFollowUpNotActiveError"
        ) {
          sendMessage({ text: trimmed });
          return;
        }
        throw queueError;
      }
      return;
    }
    sendMessage({ text: trimmed });
  }

  return {
    sessionId,
    startFailure,
    retryStart: start,
    messages: withoutKickoff(messages),
    status,
    error: isOver ? undefined : error,
    stop,
    onSend,
    queuedMessages,
    isHydrating: !!sessionId && !session,
    isTurnSettling: isReconnecting || isFinishProbing || hasActiveStream,
    isTurnRunning,
    isOver,
    canFinishEarly:
      hasAnswered(messages) && !isTurnRunning && !isReadingStage && !isOver,
    isLeaving,
    hasFinishFailed,
    finish,
    skip,
  };
}

// The server builds the team from the saved wizard answers, and a change
// reaches it only after a short delay unless flushed.
async function saveWizardAnswers() {
  await useOnboardingWizardStore
    .getState()
    .flushProgress?.()
    .catch(() => undefined);
}

async function fetchStage(sessionId: string) {
  try {
    const response = await getV2GetSession(sessionId, { limit: 1 });
    return response.status === 200
      ? (response.data.metadata?.onboarding_interview_stage ?? null)
      : null;
  } catch {
    return null;
  }
}

function pause(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}
