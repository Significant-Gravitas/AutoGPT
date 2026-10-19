import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";

import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

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
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
  },
  useGetFlag: () => false,
}));

const SYSTEM = "[__COPILOT_SYSTEM_e3b0__]";

function marker(
  sequence: number,
  pause: Record<string, unknown>,
  note: string,
) {
  return {
    id: `row-${sequence}`,
    role: "assistant",
    content: `${SYSTEM} ${note}`,
    sequence,
    metadata: {
      turn_paused: {
        resets_at: null,
        steps: null,
        checkpoint_saved: true,
        started: true,
        ...pause,
      },
    },
  };
}

function row(sequence: number, role: string, content: string) {
  return { id: `row-${sequence}`, role, content, sequence };
}

const PAUSED_CHAT = [
  row(0, "user", "Build the quarterly report."),
  row(1, "assistant", "Halfway there: the numbers are in."),
  marker(
    2,
    { reason: "steps", steps: 450 },
    "Paused: checking in after a long run.",
  ),
];

beforeEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  window.localStorage.clear();
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("the Paused card in a reopened chat", () => {
  it("is rebuilt from the stored marker row, and only the last one offers ways back in", async () => {
    renderHost({
      sessionOverride: {
        messages: [
          ...PAUSED_CHAT,
          row(3, "user", "Carry on with the charts."),
          row(4, "assistant", "Charts drafted."),
          marker(
            5,
            { reason: "task" },
            "Paused: this task reached its size limit.",
          ),
        ],
      },
    });

    const card = await screen.findByRole(
      "region",
      { name: "Paused · this task reached its size limit" },
      { timeout: 5000 },
    );
    expect(
      within(card).getByRole("button", { name: "Keep going" }),
    ).toBeDefined();
    expect(
      within(card).getByRole("button", { name: "Stop here" }),
    ).toBeDefined();
    expect(
      screen.getByRole("note", {
        name: "Paused · checking in after a long run. Picked back up",
      }),
    ).toBeDefined();
    // The row's grey system note gives way to the card it now draws.
    expect(
      screen.queryByText("Paused: this task reached its size limit."),
    ).toBeNull();
  }, 15_000);

  it("picks the task up on Keep going and follows the turn the server started", async () => {
    // The route stores the continuation note and dispatches the turn before
    // it answers; the page finds that turn on the session and resumes it.
    const note = row(
      3,
      "assistant",
      `${SYSTEM} Keep going: pick the task up from the checkpoint above.`,
    );
    const answer = "Back on it: the charts are next.";
    const continued: unknown[] = [];
    let phase: "paused" | "running" | "done" = "paused";
    server.use(
      http.post(
        "*/api/chat/sessions/:sessionId/continue",
        async ({ request }) => {
          continued.push(await request.json());
          phase = "running";
          return HttpResponse.json({ turn_id: "turn-2" }, { status: 202 });
        },
      ),
      http.get(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        ({ request }) => {
          phase = "done";
          return streamSseResponse(assistantTextChunks(answer), {
            abortSignal: request.signal,
          });
        },
      ),
    );
    renderHost({
      sessionResponse: getGetV2GetSessionMockHandler200(() =>
        session(
          phase === "paused"
            ? { messages: PAUSED_CHAT }
            : phase === "running"
              ? {
                  messages: [...PAUSED_CHAT, note],
                  active_stream: {
                    turn_id: "turn-2",
                    last_message_id: "row-3",
                    started_at: "2026-05-13T00:00:00Z",
                  },
                }
              : {
                  messages: [...PAUSED_CHAT, note, row(4, "assistant", answer)],
                },
        ),
      ),
    });

    const user = userEvent.setup();
    await user.click(
      await screen.findByRole(
        "button",
        { name: "Keep going" },
        { timeout: 5000 },
      ),
    );

    expect(
      await screen.findByText(answer, undefined, { timeout: 5000 }),
    ).toBeDefined();
    expect(continued).toEqual([{ model: "standard" }]);
    // Once the turn ends the stored rows come back, and the pause they hold
    // has the note and the new work after it: history now.
    expect(
      await screen.findByRole(
        "note",
        { name: "Paused · checking in after a long run. Picked back up" },
        { timeout: 5000 },
      ),
    ).toBeDefined();
    expect(screen.getByText(answer)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Keep going" })).toBeNull();
  }, 15_000);
});

describe("the Paused card as the turn lands", () => {
  it("stays stopped when the live pause it was stopped on comes back as its stored row", async () => {
    let sent = false;
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        ({ request }) => {
          sent = true;
          return streamSseResponse(
            [
              { type: "start", messageId: "live-turn" },
              { type: "start-step" },
              { type: "text-start", id: "t1" },
              {
                type: "text-delta",
                id: "t1",
                delta: "Halfway there: the numbers are in.",
              },
              { type: "text-end", id: "t1" },
              { type: "finish-step" },
              {
                type: "data-turn-paused",
                data: {
                  reason: "steps",
                  resets_at: null,
                  steps: 450,
                  checkpoint_saved: true,
                  started: true,
                },
              },
              { type: "finish" },
            ],
            // The turn ends a beat later, so the card is still the live part,
            // with no stored row yet, when it is stopped.
            {
              abortSignal: request.signal,
              perChunkDelaysMs: [0, 0, 0, 0, 0, 0, 0, 1500],
            },
          );
        },
      ),
    );
    renderHost({
      sessionResponse: getGetV2GetSessionMockHandler200(() =>
        session({ messages: sent ? PAUSED_CHAT : [] }),
      ),
    });

    await typeAndSend("Build the quarterly report.");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole(
        "button",
        { name: "Stop here" },
        { timeout: 5000 },
      ),
    );
    expect(screen.getByText("Stopped here")).toBeDefined();

    // Once the turn ends the chat swaps the live part for the stored row,
    // which the card then remounts from.
    await waitFor(
      () =>
        expect(
          document.querySelector(
            `[data-message-id="${TEST_SESSION_ID}-seq-2"]`,
          ),
        ).not.toBeNull(),
      { timeout: 8000 },
    );
    expect(
      screen.getByRole("note", {
        name: "Paused · checking in after a long run. Stopped here",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Keep going" })).toBeNull();
  }, 20_000);
});

function session(over: Partial<SessionDetailResponse>): SessionDetailResponse {
  return {
    id: TEST_SESSION_ID,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    user_id: "test-user",
    chat_status: "idle",
    messages: [],
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: null,
    metadata: { dry_run: false, builder_graph_id: null },
    expert_id: null,
    ...over,
  };
}
