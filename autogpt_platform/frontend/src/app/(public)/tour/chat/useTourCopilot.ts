"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";
import { appendPartToLastMessage, textRevealDurationMs } from "./helpers";
import type { TourMessage, TourScript } from "./script/types";

type TourStatus = "ready" | "streaming";

/** Beat after the turn's last text finishes revealing before the next prompt
 * prefills — keeps the visitor's attention on the transcript, not the bar. */
const TURN_SETTLE_MS = 1500;
/** Hold after the final part streams in before the demo flips to the upsell
 * card — gives the visitor time to take in the payoff artifact. */
const FINAL_TURN_SETTLE_MS = 3000;
const AUTO_START_DELAY_MS = 1000;

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

    // A manual send cancels the pending auto-start so the turn can't fire twice.
    clearTimers();

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
    // Text parts keep typing themselves out after they commit, so the turn is
    // only visually over once the slowest reveal finishes — not at last commit.
    let visualEnd = 0;
    turn.steps.forEach((step) => {
      elapsed += step.delayMs;
      timers.current.push(
        setTimeout(() => {
          commit(appendPartToLastMessage(messagesRef.current, step.part));
        }, elapsed),
      );
      const revealMs =
        step.part.type === "text" ? textRevealDurationMs(step.part.text) : 0;
      visualEnd = Math.max(visualEnd, elapsed + revealMs);
    });

    const isFinalTurn = stepIndex.current === script.length - 1;
    timers.current.push(
      setTimeout(
        () => {
          setStatus("ready");
          stepIndex.current += 1;
          if (stepIndex.current >= script.length) onComplete();
        },
        visualEnd + (isFinalTurn ? FINAL_TURN_SETTLE_MS : TURN_SETTLE_MS),
      ),
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
    onReset?.();
    scheduleAutoStart();
  }

  // TourChatHost is keyed by scenario id, so a scenario switch remounts this
  // hook — each fresh mount auto-plays its first turn after a short beat
  // (pressing Enter still works and just skips the wait).
  useMountEffect(() => {
    scheduleAutoStart();
    return clearTimers;
  });

  const currentTurn = script[stepIndex.current];
  // The first turn auto-plays, so the prompt bar stays empty and disabled
  // until it finishes — only then does the next turn's prompt prefill.
  const currentUserPrompt =
    stepIndex.current === 0 ? null : (currentTurn?.userPrompt ?? null);

  return {
    messages,
    onSend,
    reset,
    turnIndex: stepIndex.current,
    currentUserPrompt,
    isStreaming: status === "streaming",
    isExhausted: stepIndex.current >= script.length,
  };
}
