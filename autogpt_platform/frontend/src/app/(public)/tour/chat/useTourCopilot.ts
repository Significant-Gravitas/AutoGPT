"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";
import { appendPartToLastMessage } from "./helpers";
import type { TourMessage, TourScript } from "./script/types";

type TourStatus = "ready" | "streaming";

const TURN_SETTLE_MS = 300;
/** Hold after the final part streams in before the demo flips to the upsell
 * card — gives the visitor time to take in the payoff artifact. */
const FINAL_TURN_SETTLE_MS = 3000;

interface Args {
  sessionId: string;
  script: TourScript;
  onComplete: () => void;
  onReset?: () => void;
}

export function useTourCopilot({
  sessionId,
  script,
  onComplete,
  onReset,
}: Args) {
  const [messages, setMessages] = useState<TourMessage[]>([]);
  const [status, setStatus] = useState<TourStatus>("ready");
  const stepIndex = useRef(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const messagesRef = useRef<TourMessage[]>([]);

  function commit(next: TourMessage[]) {
    messagesRef.current = next;
    setMessages(next);
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }

  function onSend(text: string) {
    const turn = script[stepIndex.current];
    if (status !== "ready" || !turn) return;

    commit([
      ...messagesRef.current,
      {
        id: `${sessionId}-user-${stepIndex.current}`,
        role: "user",
        parts: [{ type: "text", text }],
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

    const isFinalTurn = stepIndex.current === script.length - 1;
    timers.current.push(
      setTimeout(
        () => {
          setStatus("ready");
          stepIndex.current += 1;
          if (stepIndex.current >= script.length) onComplete();
        },
        elapsed + (isFinalTurn ? FINAL_TURN_SETTLE_MS : TURN_SETTLE_MS),
      ),
    );
  }

  function reset() {
    clearTimers();
    stepIndex.current = 0;
    commit([]);
    setStatus("ready");
    onReset?.();
  }

  useMountEffect(() => clearTimers);

  const currentTurn = script[stepIndex.current];

  return {
    messages,
    onSend,
    reset,
    turnIndex: stepIndex.current,
    currentUserPrompt: currentTurn?.userPrompt ?? null,
    isStreaming: status === "streaming",
    isExhausted: stepIndex.current >= script.length,
  };
}
