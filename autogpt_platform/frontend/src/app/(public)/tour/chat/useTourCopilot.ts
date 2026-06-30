"use client";

import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useMountEffect } from "@/hooks/useMountEffect";
import type { UIMessage } from "ai";
import { useRef, useState } from "react";
import { appendPartToLastMessage } from "./helpers";
import { mockArtifact } from "./script/mockArtifact";
import { monitorPricingScript } from "./script/monitorPricingScript";
import { TOUR_SESSION_ID } from "./script/mockSidebarSessions";

type TourStatus = "ready" | "submitted" | "streaming";

export function useTourCopilot({ onComplete }: { onComplete: () => void }) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [status, setStatus] = useState<TourStatus>("ready");
  const stepIndex = useRef(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const messagesRef = useRef<UIMessage[]>([]);

  function commit(next: UIMessage[]) {
    messagesRef.current = next;
    setMessages(next);
    useCopilotStreamStore.getState().setMessageSnapshot(TOUR_SESSION_ID, next);
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }

  function onSend(text: string) {
    const turn = monitorPricingScript[stepIndex.current];
    if (status !== "ready" || !turn) return;

    commit([
      ...messagesRef.current,
      {
        id: `tour-user-${stepIndex.current}`,
        role: "user",
        parts: [{ type: "text", text, state: "done" }],
      },
      { id: turn.assistantMessageId, role: "assistant", parts: [] },
    ]);
    setStatus("streaming");

    let elapsed = 0;
    turn.steps.forEach((step) => {
      elapsed += step.delayMs;
      timers.current.push(
        setTimeout(() => {
          commit(appendPartToLastMessage(messagesRef.current, step.part));
          if (step.part.type === "tool-create_agent") {
            useCopilotUIStore.getState().openArtifact(mockArtifact);
          }
        }, elapsed),
      );
    });

    timers.current.push(
      setTimeout(() => {
        setStatus("ready");
        stepIndex.current += 1;
        if (stepIndex.current >= monitorPricingScript.length) onComplete();
      }, elapsed + 300),
    );
  }

  function reset() {
    clearTimers();
    stepIndex.current = 0;
    commit([]);
    setStatus("ready");
  }

  useMountEffect(() => clearTimers);

  return {
    sessionId: TOUR_SESSION_ID,
    messages,
    status,
    onSend,
    reset,
    error: undefined,
    stop: () => {},
    isReconnecting: false,
    isRestoringActiveSession: false,
    restoreStatusMessage: null,
    activeStreamStartedAt: null,
    isUserStopping: false,
    createSession: () => {},
    onEnqueue: () => {},
    queuedMessages: [] as string[],
    isLoadingSession: false,
    isSessionError: false,
    isCreatingSession: false,
    isUploadingFiles: false,
    hasMoreMessages: false,
    isLoadingMore: false,
    loadMore: () => {},
    turnStats: new Map(),
    rateLimitMessage: null,
    dismissRateLimit: () => {},
    sessionDryRun: false,
    sessionChatStatus: "idle",
  };
}
