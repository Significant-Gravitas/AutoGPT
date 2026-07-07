"use client";

import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import { useMountEffect } from "@/hooks/useMountEffect";
import type { UIMessage } from "ai";
import { useRef, useState } from "react";
import { appendPartToLastMessage } from "./helpers";
import type { TourScript } from "./script/types";

type TourStatus = "ready" | "submitted" | "streaming";

const AUTO_START_DELAY_MS = 1000;

interface Args {
  sessionId: string;
  script: TourScript;
  onComplete: () => void;
}

export function useTourCopilot({ sessionId, script, onComplete }: Args) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [status, setStatus] = useState<TourStatus>("ready");
  const stepIndex = useRef(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const messagesRef = useRef<UIMessage[]>([]);

  function commit(next: UIMessage[]) {
    messagesRef.current = next;
    setMessages(next);
    useCopilotStreamStore.getState().setMessageSnapshot(sessionId, next);
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }

  function onSend(text: string) {
    const turn = script[stepIndex.current];
    if (status !== "ready" || !turn) return;

    // A manual send cancels the pending auto-start so the turn can't fire twice.
    clearTimers();

    commit([
      ...messagesRef.current,
      {
        id: `${sessionId}-user-${stepIndex.current}`,
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
        }, elapsed),
      );
    });

    timers.current.push(
      setTimeout(() => {
        setStatus("ready");
        stepIndex.current += 1;
        if (stepIndex.current >= script.length) onComplete();
      }, elapsed + 300),
    );
  }

  function scheduleAutoStart() {
    timers.current.push(
      setTimeout(() => {
        const turn = script[stepIndex.current];
        if (turn) onSend(turn.userPrompt);
      }, AUTO_START_DELAY_MS),
    );
  }

  function reset() {
    clearTimers();
    stepIndex.current = 0;
    commit([]);
    setStatus("ready");
    scheduleAutoStart();
  }

  // Fresh slate when this chat mounts. TourChatHost is keyed by sessionId, so a
  // sidebar switch remounts this hook — clear any prior snapshot so the messages
  // reflect the newly-selected chat, then auto-play its first turn after a short
  // beat (pressing Enter still works and just skips the wait).
  useMountEffect(() => {
    useCopilotStreamStore.getState().setMessageSnapshot(sessionId, []);
    scheduleAutoStart();
    return clearTimers;
  });

  const currentTurn = script[stepIndex.current];

  return {
    sessionId,
    messages,
    status,
    error: undefined as Error | undefined,
    turnStats: new Map(),
    queuedMessages: [] as string[],
    onSend,
    reset,
    turnIndex: stepIndex.current,
    currentUserPrompt: currentTurn?.userPrompt ?? null,
    isStreaming: status === "streaming" || status === "submitted",
    isExhausted: stepIndex.current >= script.length,
  };
}
