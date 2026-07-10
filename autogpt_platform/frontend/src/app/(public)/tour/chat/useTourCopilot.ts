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
/** Beat before the first prompt starts typing itself into the bar. */
const AUTO_START_DELAY_MS = 1000;
/** Pause between the auto-typed prompt finishing and its "Enter press". */
const AUTO_SEND_DELAY_MS = 600;
/** Beat after the final turn's last part before the artifact heads-up types in. */
const COMPLETION_NOTICE_DELAY_MS = 600;

interface Args {
  sessionId: string;
  script: TourScript;
  onComplete: () => void;
  /** Extra closing line for the final turn, pointing at the artifact panel
   * that is about to open (e.g. "Your report.md will appear on the right"). */
  completionNotice?: string;
}

export function useTourCopilot({
  sessionId,
  script,
  onComplete,
  completionNotice,
}: Args) {
  const [messages, setMessages] = useState<TourMessage[]>([]);
  const [status, setStatus] = useState<TourStatus>("ready");
  const [isAutoPromptVisible, setIsAutoPromptVisible] = useState(false);
  const stepIndex = useRef(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const messagesRef = useRef<TourMessage[]>([]);
  // onSend runs from setTimeout callbacks (auto-start), so it must read status
  // through a ref — the closure's `status` is stale by the time a timer fires.
  const statusRef = useRef<TourStatus>("ready");

  function commit(next: TourMessage[]) {
    messagesRef.current = next;
    setMessages(next);
  }

  function transition(next: TourStatus) {
    statusRef.current = next;
    setStatus(next);
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }

  function onSend(text: string) {
    const turn = script[stepIndex.current];
    if (statusRef.current !== "ready" || !turn) return;

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
    transition("streaming");

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
    if (isFinalTurn && completionNotice) {
      const noticeAt = visualEnd + COMPLETION_NOTICE_DELAY_MS;
      timers.current.push(
        setTimeout(() => {
          commit(
            appendPartToLastMessage(messagesRef.current, {
              type: "text",
              text: completionNotice,
            }),
          );
        }, noticeAt),
      );
      visualEnd = noticeAt + textRevealDurationMs(completionNotice);
    }
    timers.current.push(
      setTimeout(
        () => {
          transition("ready");
          stepIndex.current += 1;
          if (stepIndex.current >= script.length) onComplete();
        },
        visualEnd + (isFinalTurn ? FINAL_TURN_SETTLE_MS : TURN_SETTLE_MS),
      ),
    );
  }

  // First turn plays itself like a real interaction: after a beat the prompt
  // types into the bar (the reveal runs in TourPromptBar at a shared speed, so
  // its end time is known here), then it "presses Enter" on its own.
  function scheduleAutoStart() {
    const firstTurn = script[0];
    if (!firstTurn) return;
    timers.current.push(
      setTimeout(() => setIsAutoPromptVisible(true), AUTO_START_DELAY_MS),
    );
    timers.current.push(
      setTimeout(
        () => onSend(firstTurn.userPrompt),
        AUTO_START_DELAY_MS +
          textRevealDurationMs(firstTurn.userPrompt) +
          AUTO_SEND_DELAY_MS,
      ),
    );
  }

  // TourChatHost is keyed by scenario id, so a scenario switch remounts this
  // hook — each fresh mount auto-plays its first turn (pressing Enter still
  // works and just skips the wait).
  useMountEffect(() => {
    scheduleAutoStart();
    return clearTimers;
  });

  const currentTurn = script[stepIndex.current];
  // Turn 0's prompt is only in the bar while it auto-types; it clears on send
  // and the next turn's prompt prefills once the round finishes.
  const currentUserPrompt =
    stepIndex.current === 0
      ? status === "ready" && isAutoPromptVisible
        ? (currentTurn?.userPrompt ?? null)
        : null
      : (currentTurn?.userPrompt ?? null);

  return {
    messages,
    onSend,
    turnIndex: stepIndex.current,
    currentUserPrompt,
    isStreaming: status === "streaming",
  };
}
