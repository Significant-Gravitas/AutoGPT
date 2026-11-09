import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import {
  finishOnboardingInterview,
  getOnboardingInterview,
  startOnboardingInterview,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import {
  getV2GetSession,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { trackInterview } from "../../tracking";
import { recordSkippedTake } from "../BrainDumpStep/useBrainDumpStep";
import { useMicrophonePermission } from "../BrainDumpStep/useMicrophonePermission";
import { forgetShownPlan } from "../SubscriptionStep/components/PlanPanel/helpers";
import {
  type MicrophoneRefusal,
  microphoneRefusalOf,
  OPENING_READ_TIMEOUT_MS,
} from "./components/VoiceCall/helpers";
import { useVoiceCall } from "./components/VoiceCall/useVoiceCall";
import {
  finishFailureOf,
  finishPathOf,
  forgetTyping,
  type HeldBy,
  hasAnswered,
  isEndedStage,
  isInterviewClosedError,
  isTypingRemembered,
  refusalOf,
  refusedMessageOf,
  rememberTyping,
  STAGE_READ_INTERVAL_MS,
  STAGE_READ_WINDOW_MS,
  type StartFailure,
  startFailureOf,
  turnRefusalOf,
  withoutKickoff,
} from "./helpers";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

// How the interview runs on the step: the choice between a call and typing,
// the call, or the typed chat.
export type InterviewMode = "choose" | "talk" | "type";

export function useInterviewStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const setStepBusy = useOnboardingWizardStore((s) => s.setStepBusy);
  const userID = useOnboardingWizardStore((s) => s.userID);
  const voiceFlag = useFlagStatus(Flag.ONBOARDING_INTERVIEW_VOICE);
  const microphone = useMicrophonePermission();

  // What the person, or a server's answer, put on the step: the typed chat
  // or the call. Until then the opening below decides.
  const [surface, setSurface] = useState<"type" | "voice" | null>(null);
  const [micRefusal, setMicRefusal] = useState<MicrophoneRefusal | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [startFailure, setStartFailure] = useState<StartFailure | null>(null);
  const [startedStage, setStartedStage] = useState<string | null>(null);
  const [readStage, setReadStage] = useState<string | null>(null);
  const [hasAskedToFinish, setHasAskedToFinish] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  const [finishFailure, setFinishFailure] = useState<StartFailure | null>(null);
  const isLeavingRef = useRef(false);
  const handledRefusalRef = useRef<Error | undefined>(undefined);
  const startedRef = useRef(false);
  const startAttemptRef = useRef(0);
  const turnRanRef = useRef(false);

  // With the voice flag on, the step opens on the person's latest call,
  // read before anything typed (useVoiceCall); with no call to pick up, the
  // microphone and a choice to type made earlier in this tab decide.
  const opening = !voiceFlag.ready
    ? null
    : voiceFlag.enabled
      ? "voice"
      : "type";
  const shown = surface ?? opening;

  const voice = useVoiceCall({
    userId: userID,
    isOffered: shown === "voice" && voiceFlag.enabled,
    onTypeInstead: openTyped,
    onLeave: (path) => leaveToTeam(path, "voice"),
    onSkip: skip,
    onCallStarted: () => forgetTyping(userID),
    onMicrophoneRefused: refuseMicrophone,
    onNoCall: openWithoutCall,
  });

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
  // A typed turn refused because a call holds the interview: shown as that
  // call, never as an error on the chat.
  const turnRefusal = turnRefusalOf(error);
  const isTypedOver =
    hasAskedToFinish ||
    isEndedStage(startedStage) ||
    isEndedStage(session?.metadata?.onboarding_interview_stage) ||
    isEndedStage(readStage) ||
    (isInterviewClosedError(error) && !turnRefusal);
  // Either mode's interview is over: the typed one closed, or a call took
  // it.
  const isOver = shown === "voice" ? voice.isOver : isTypedOver;

  const mode: InterviewMode | null =
    shown === null
      ? null
      : shown === "type"
        ? "type"
        : voice.phase === "idle" || voice.phase === "checking"
          ? "choose"
          : "talk";

  // Leaving, a call in flight, or a call's finish on its way: Back stays
  // out of the way.
  const isStepBusy = isLeaving || voice.isBusy || voice.isLeaving;
  useEffect(() => {
    setStepBusy(isStepBusy);
    return () => setStepBusy(false);
  }, [isStepBusy, setStepBusy]);

  // Otto's end signal closes the interview inside the turn; the reply cap and
  // the scope guard close it just after. So once a turn ends the stage is read
  // for a while.
  useEffect(() => {
    if (isTurnRunning) {
      turnRanRef.current = true;
      return;
    }
    if (!turnRanRef.current || !sessionId) return;
    turnRanRef.current = false;
    let cancelled = false;
    async function readUntilOver(id: string) {
      const deadline = Date.now() + STAGE_READ_WINDOW_MS;
      while (!cancelled && Date.now() < deadline) {
        const stage = await fetchStage(id);
        if (cancelled) return;
        if (isEndedStage(stage)) {
          setReadStage(stage);
          return;
        }
        await pause(STAGE_READ_INTERVAL_MS);
      }
    }
    void readUntilOver(sessionId);
    return () => {
      cancelled = true;
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
      const heldBy = refusalOf(startError).heldBy;
      if (heldBy?.callId) {
        showCallThatHolds(heldBy);
        return;
      }
      setStartFailure(startFailureOf(startError));
    }
  }

  // A voice call holds the interview: rejoin or end the live one, or show
  // how the one that took the interview ended.
  function showCallThatHolds(heldBy: HeldBy) {
    if (!heldBy.callId) return;
    setSurface("voice");
    if (heldBy.kind === "call_in_progress") {
      voice.showHeld(heldBy.callId, "type");
    } else {
      voice.showTakenBy(heldBy.callId);
    }
  }

  // No call to pick up. Before the choice and its Skip can show, the typed
  // interview is read: one already over opens as its transcript with "Meet
  // your team", since a Skip would record a take over it. A refused
  // microphone, or a choice to type made earlier in this tab, opens the
  // typed interview too; otherwise the choice. The permission is read now,
  // not from the last render.
  async function openWithoutCall() {
    const [typed, permission] = await Promise.all([
      readTypedInterview(),
      microphone.refresh(),
    ]);
    if (
      typed?.is_over ||
      permission === "denied" ||
      isTypingRemembered(userID)
    ) {
      openTyped({ remember: false });
    }
  }

  // Puts the typed interview on the step, opening it again when a call held
  // it before.
  function openTyped({ remember }: { remember: boolean }) {
    if (remember) rememberTyping(userID);
    setSurface("type");
    if (startedRef.current) void start();
  }

  function refuseMicrophone(refusal: MicrophoneRefusal) {
    setMicRefusal(refusal);
    openTyped({ remember: false });
    // A dismissed prompt fires no change event, so the state is read again
    // for the notice to say which refusal it was.
    void microphone.refresh();
  }

  // After a refused microphone: ask the browser again, and offer the call
  // once it says yes.
  async function retryMicrophone() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      stream.getTracks().forEach((track) => track.stop());
    } catch (micError) {
      setMicRefusal(microphoneRefusalOf(micError));
      await microphone.refresh();
      return;
    }
    setMicRefusal(null);
    forgetTyping(userID);
    setSurface("voice");
    await microphone.refresh();
  }

  async function finish() {
    if (isLeavingRef.current) return;
    isLeavingRef.current = true;
    setIsLeaving(true);
    setHasAskedToFinish(true);
    setFinishFailure(null);
    let path: "A" | "B" | null = null;
    let failure: StartFailure | null = null;
    try {
      const response = await finishOnboardingInterview();
      path = response.status === 200 ? finishPathOf(response.data) : null;
    } catch (finishError) {
      // Refused because a voice call holds the interview: the call shows,
      // as it does for the typed start.
      const heldBy = refusalOf(finishError).heldBy;
      if (heldBy?.callId) {
        isLeavingRef.current = false;
        setIsLeaving(false);
        showCallThatHolds(heldBy);
        return;
      }
      failure = finishFailureOf(finishError);
    }
    if (path === null) {
      isLeavingRef.current = false;
      setIsLeaving(false);
      setFinishFailure(failure ?? { kind: "failed" });
      return;
    }
    await leaveToTeam(path, "text");
  }

  // Both modes leave the step the same way, on the path the hand-off's
  // answer decided.
  async function leaveToTeam(path: "A" | "B", leftFrom: "text" | "voice") {
    isLeavingRef.current = true;
    setIsLeaving(true);
    setIntroPath(path);
    forgetShownPlan(useOnboardingWizardStore.getState().userID);
    trackInterview("team_screen_reached", { mode: leftFrom, path });
    await saveWizardAnswers();
    nextStep();
  }

  async function skip() {
    if (isOver || isStepBusy || isLeavingRef.current) return;
    isLeavingRef.current = true;
    setIsLeaving(true);
    setIntroPath("B");
    forgetShownPlan(useOnboardingWizardStore.getState().userID);
    await saveWizardAnswers();
    nextStep();
    await recordSkippedTake();
  }

  // The server never took the refused message: it leaves the chat, and its
  // text waits in the composer for when the person types again.
  function takeBackRefusedMessage() {
    const refused = refusedMessageOf(messages);
    if (!refused) return;
    setMessages(messages.filter((message) => message.id !== refused.id));
    if (refused.text) {
      useCopilotUIStore.getState().setInitialPrompt(refused.text);
    }
  }

  // Before the person's first message after a call, the call's words can
  // still land and take the interview; the turn refused for it shows that
  // call's ending, as the typed start's refusal does.
  useEffect(() => {
    if (!turnRefusal || handledRefusalRef.current === error) return;
    handledRefusalRef.current = error;
    takeBackRefusedMessage();
    showCallThatHolds(turnRefusal);
  });

  // The typed interview opens once it is on the step; a retry calls start
  // itself.
  useEffect(() => {
    if (shown !== "type" || startedRef.current) return;
    startedRef.current = true;
    void start();
  });

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
    mode,
    voice,
    isMicBlocked:
      voiceFlag.enabled &&
      (micRefusal === "blocked" || microphone.state === "denied"),
    isMicUnavailable: micRefusal === "unavailable",
    micPermission: microphone.state,
    retryMicrophone,
    sessionId,
    startFailure,
    retryStart: start,
    messages: withoutKickoff(messages),
    status,
    error: isOver || turnRefusal ? undefined : error,
    stop,
    onSend,
    queuedMessages,
    isHydrating: !!sessionId && !session,
    isTurnSettling: isReconnecting || isFinishProbing || hasActiveStream,
    isTurnRunning,
    isOver,
    // Skip is for an open interview, and keeps out of the way of a call in
    // flight, whose words would land after it.
    canSkip: !isOver && !isStepBusy,
    canFinishEarly: hasAnswered(messages) && !isTurnRunning && !isOver,
    isLeaving,
    finishFailure,
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

// The typed interview as it stands, read without starting it; null when the
// person has none or it cannot be read, which leaves the choice to show.
async function readTypedInterview() {
  const giveUp = new AbortController();
  const timer = setTimeout(() => giveUp.abort(), OPENING_READ_TIMEOUT_MS);
  try {
    const response = await getOnboardingInterview({ signal: giveUp.signal });
    return response.status === 200 ? response.data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
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
