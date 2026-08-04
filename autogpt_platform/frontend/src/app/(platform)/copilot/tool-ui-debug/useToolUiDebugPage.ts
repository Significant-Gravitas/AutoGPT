import { useRef, useState } from "react";
import { applyEvent, type ChatMessage } from "./helpers";
import { buildSampleEvents } from "./sampleScript";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export type ToolUiVariant = "new" | "old";

export function useToolUiDebugPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<"ready" | "streaming">("ready");
  const [awaitingUser, setAwaitingUser] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [variant, setVariant] = useState<ToolUiVariant>("new");
  const runRef = useRef(0);
  const answerRef = useRef<((text: string) => void) | null>(null);
  const userCountRef = useRef(0);

  function appendUserMessage(text: string) {
    userCountRef.current += 1;
    setMessages((prev) => [
      ...prev,
      {
        id: `debug-user-${userCountRef.current}`,
        role: "user",
        parts: [{ type: "text", text }],
      },
    ]);
  }

  async function play() {
    const runId = ++runRef.current;
    setMessages([]);
    setStatus("streaming");
    setAwaitingUser(false);
    setStatusMessage(null);
    for (const event of buildSampleEvents()) {
      await sleep(event.delay);
      if (runRef.current !== runId) return;
      if (event.kind === "status") {
        setStatusMessage(event.message);
        continue;
      }
      setStatusMessage(null);
      if (event.kind === "await-user") {
        setStatus("ready");
        setAwaitingUser(true);
        const answer = await new Promise<string>((resolve) => {
          answerRef.current = resolve;
        });
        if (runRef.current !== runId) return;
        setAwaitingUser(false);
        if (answer.trim()) appendUserMessage(answer);
        setStatus("streaming");
        continue;
      }
      setMessages((prev) => applyEvent(prev, event));
    }
    if (runRef.current === runId) setStatus("ready");
  }

  // Interactive cards (ask_question, setup cards) call this via
  // CopilotChatActionsProvider.onSend. If playback is paused on an
  // await-user event, the answer resumes it; otherwise the message just
  // lands in the transcript.
  function sendUserMessage(text: string) {
    const resume = answerRef.current;
    if (resume) {
      answerRef.current = null;
      resume(text);
      return;
    }
    appendUserMessage(text);
  }

  function reset() {
    runRef.current += 1;
    const resume = answerRef.current;
    answerRef.current = null;
    resume?.("");
    setMessages([]);
    setStatus("ready");
    setAwaitingUser(false);
    setStatusMessage(null);
  }

  return {
    messages,
    status,
    isPlaying: status === "streaming",
    awaitingUser,
    statusMessage,
    variant,
    setVariant,
    play,
    reset,
    sendUserMessage,
  };
}
