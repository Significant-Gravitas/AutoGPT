import { server } from "@/mocks/mock-server";
import { streamSseResponse } from "@/tests/integrations/copilot-sse";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// Pin the backend host so the CoPilot transport's absolute URL is deterministic.
vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return { ...actual, getCopilotAuthHeaders: async () => ({}) };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

// Only the chips flag is on: everything else off keeps the input rendering a
// single Submit button, which ``typeAndSend`` relies on.
const flagState = vi.hoisted(() => ({ nextStepChips: true }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.COPILOT_NEXT_STEP_CHIPS
        ? flagState.nextStepChips
        : false,
  };
});

const SUGGESTIONS = [
  "Email the report",
  "Post on r/SaaS",
  "Fix the criticals",
];

function turnWithSuggestions(
  text: string,
  suggestions: string[],
  messageId: string,
): UIMessageChunk[] {
  const textPartId = `${messageId}-text`;
  return [
    { type: "start", messageId },
    { type: "start-step" },
    { type: "text-start", id: textPartId },
    { type: "text-delta", id: textPartId, delta: text },
    { type: "text-end", id: textPartId },
    { type: "data-suggestions", data: { suggestions } },
    { type: "finish-step" },
    { type: "finish" },
  ] as UIMessageChunk[];
}

/**
 * Records the ``message`` field of every stream POST so a test can assert
 * what the chip actually sent, not just what re-rendered.
 */
function recordingStreamHandler(chunksPerTurn: UIMessageChunk[][]) {
  const sentMessages: string[] = [];
  let turn = 0;
  const handler = http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
    async ({ request }) => {
      const body = (await request.clone().json()) as { message?: string };
      sentMessages.push(body.message ?? "");
      const chunks =
        chunksPerTurn[turn] ?? chunksPerTurn[chunksPerTurn.length - 1];
      turn += 1;
      return streamSseResponse(chunks, { abortSignal: request.signal });
    },
  );
  return { handler, sentMessages };
}

beforeEach(() => {
  flagState.nextStepChips = true;
  resetCopilotChatRegistry();
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("next-step chips", () => {
  it("renders the model's suggestions as chips under the final assistant message", async () => {
    const { handler } = recordingStreamHandler([
      turnWithSuggestions("Report is ready.", SUGGESTIONS, "assistant-1"),
    ]);
    server.use(handler);
    renderHost();

    await typeAndSend("build me a report");

    expect(
      await screen.findByText("Report is ready.", undefined, { timeout: 5000 }),
    ).toBeDefined();

    const chips = await screen.findByTestId("next-step-chips");
    for (const label of SUGGESTIONS) {
      expect(within(chips).getByRole("button", { name: label })).toBeDefined();
    }

    // The chips belong to the assistant message that produced them, not to
    // the message list as a whole.
    const assistantMessage = chips.closest("[data-message-id]");
    expect(assistantMessage).not.toBeNull();
    expect(
      within(assistantMessage as HTMLElement).getByText("Report is ready."),
    ).toBeDefined();
  });

  it("sends the chip's text as the next user message when clicked", async () => {
    const { handler, sentMessages } = recordingStreamHandler([
      turnWithSuggestions("Report is ready.", SUGGESTIONS, "assistant-1"),
      turnWithSuggestions("Sent it over.", [], "assistant-2"),
    ]);
    server.use(handler);
    renderHost();

    await typeAndSend("build me a report");

    const chips = await screen.findByTestId("next-step-chips");
    const user = userEvent.setup();
    await user.click(within(chips).getByRole("button", { name: SUGGESTIONS[0] }));

    await waitFor(
      () => {
        expect(sentMessages).toEqual(["build me a report", "Email the report"]);
      },
      { timeout: 5000 },
    );

    expect(
      await screen.findByText("Sent it over.", undefined, { timeout: 5000 }),
    ).toBeDefined();
  });

  it("renders no chips when the model offered none", async () => {
    const { handler } = recordingStreamHandler([
      turnWithSuggestions("Yes, that's right.", [], "assistant-1"),
    ]);
    server.use(handler);
    renderHost();

    await typeAndSend("is 2 + 2 four?");

    expect(
      await screen.findByText("Yes, that's right.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(screen.queryByTestId("next-step-chips")).toBeNull();
  });

  it("renders no chips when the flag is off, even if the backend sent some", async () => {
    flagState.nextStepChips = false;
    const { handler } = recordingStreamHandler([
      turnWithSuggestions("Report is ready.", SUGGESTIONS, "assistant-1"),
    ]);
    server.use(handler);
    renderHost();

    await typeAndSend("build me a report");

    expect(
      await screen.findByText("Report is ready.", undefined, { timeout: 5000 }),
    ).toBeDefined();
    expect(screen.queryByTestId("next-step-chips")).toBeNull();
    expect(screen.queryByRole("button", { name: SUGGESTIONS[0] })).toBeNull();
  });

  it("never leaks the raw data-suggestions payload into the message body", async () => {
    const { handler } = recordingStreamHandler([
      turnWithSuggestions("Report is ready.", SUGGESTIONS, "assistant-1"),
    ]);
    server.use(handler);
    renderHost();

    await typeAndSend("build me a report");

    const chips = await screen.findByTestId("next-step-chips");
    const assistantMessage = chips.closest("[data-message-id]") as HTMLElement;
    expect(assistantMessage.textContent).not.toContain("data-suggestions");
  });
});
