import { screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { UIMessage } from "ai";
import { expect } from "vitest";
import { useCopilotStream } from "../useCopilotStream";

export const TEST_BACKEND_BASE_URL = "http://localhost:18006";
export const TEST_SESSION_ID = "test-session-stream-1";

interface Props {
  hasActiveStream?: boolean;
  hydratedMessages?: UIMessage[];
}

export function StreamHarness({
  hasActiveStream = false,
  hydratedMessages = [],
}: Props) {
  const refetchSession = async () => ({ data: undefined });
  const result = useCopilotStream({
    sessionId: TEST_SESSION_ID,
    hydratedMessages,
    hasActiveStream,
    refetchSession,
    copilotMode: undefined,
    copilotModel: undefined,
  });

  const assistantMessages = result.messages.filter(
    (m) => m.role === "assistant",
  );
  const userMessages = result.messages.filter((m) => m.role === "user");

  const allParts = assistantMessages.flatMap((m) => m.parts);

  const assistantText = allParts
    .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");

  const assistantReasoning = allParts
    .filter(
      (p): p is Extract<typeof p, { type: "reasoning" }> =>
        p.type === "reasoning",
    )
    .map((p) => p.text)
    .join("");

  interface ToolPartSnapshot {
    type: string;
    state?: string;
    toolCallId?: string;
    input?: unknown;
    output?: unknown;
  }
  const toolParts: ToolPartSnapshot[] = allParts
    .filter(
      (p) =>
        typeof p.type === "string" &&
        (p.type.startsWith("tool-") || p.type === "dynamic-tool"),
    )
    .map((p) => {
      const tp = p as ToolPartSnapshot;
      return {
        type: tp.type,
        state: tp.state,
        toolCallId: tp.toolCallId,
        input: tp.input,
        output: tp.output,
      };
    });

  const stepStartCount = allParts.filter((p) => p.type === "step-start").length;

  return (
    <div>
      <div data-testid="status">{result.status}</div>
      <div data-testid="error">{result.error?.message ?? ""}</div>
      <div data-testid="is-reconnecting">{String(result.isReconnecting)}</div>
      <div data-testid="is-user-stopping">{String(result.isUserStopping)}</div>
      <div data-testid="rate-limit">{result.rateLimitMessage ?? ""}</div>
      <div data-testid="assistant-text">{assistantText}</div>
      <div data-testid="assistant-reasoning">{assistantReasoning}</div>
      <div data-testid="tool-parts">{JSON.stringify(toolParts)}</div>
      <div data-testid="assistant-message-count">
        {assistantMessages.length}
      </div>
      <div data-testid="user-message-count">{userMessages.length}</div>
      <div data-testid="step-start-count">{stepStartCount}</div>
      <button
        data-testid="send"
        onClick={() => result.sendMessage({ text: "hello copilot" })}
      >
        send
      </button>
      <button data-testid="stop" onClick={() => result.stop()}>
        stop
      </button>
    </div>
  );
}

export async function clickSend() {
  const user = userEvent.setup();
  await user.click(screen.getByTestId("send"));
}

export async function clickStop() {
  const user = userEvent.setup();
  await user.click(screen.getByTestId("stop"));
}

export function expectTextContent(testId: string, expected: string) {
  expect(screen.getByTestId(testId).textContent).toBe(expected);
}
