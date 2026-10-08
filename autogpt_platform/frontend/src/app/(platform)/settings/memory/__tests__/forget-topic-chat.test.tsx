import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSessionMockHandler200,
  getGetV2GetSessionResponseMock200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { server } from "@/mocks/mock-server";
import { streamSseResponse } from "@/tests/integrations/copilot-sse";

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

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    GRAPHITI_MEMORY: "graphiti-memory",
    HIRE_EXPERTS: "hire-experts",
    CHAT_MODE_OPTION: "chat-mode-option",
    ENABLE_PLATFORM_PAYMENT: "enable-platform-payment",
  },
  useGetFlag: (flag: string) =>
    flag === "graphiti-memory" || flag === "hire-experts",
}));

vi.mock("@/services/feature-flags/with-feature-flag", () => ({
  withFeatureFlag: (Component: React.ComponentType) => Component,
}));

import SettingsMemoryPage from "../page";

const SESSION_ID = "forget-topic-session";

const ASK_WHICH_MEMORIES: UIMessageChunk[] = [
  { type: "start", messageId: "forget-reply-1" },
  { type: "start-step" },
  {
    type: "tool-input-available",
    toolCallId: "call-ask-forget",
    toolName: "ask_question",
    input: {},
  },
  {
    type: "tool-output-available",
    toolCallId: "call-ask-forget",
    output: {
      type: "agent_builder_clarification_needed",
      message: "Which of these should I forget?",
      questions: [
        {
          question: "Which memories should I forget?",
          keyword: "memories",
          options: ["Runs Emberline", "Monday summary emails"],
        },
      ],
    },
  },
  { type: "finish-step" },
  { type: "finish" },
];

beforeEach(() => {
  resetCopilotChatRegistry();
  window.history.replaceState({}, "", "/settings/memory");
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200({
      expert_id: null,
      items: [
        {
          uuid: "edge-1",
          fact: "Runs a DTC candle brand called Emberline",
          name: "runs",
          source: "User",
          target: "Emberline",
          created_at: "2026-08-16T00:00:00Z",
        },
      ],
    }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 1,
      entities: 1,
      episodes: 1,
    }),
    getPostV2CreateSessionMockHandler200(
      getPostV2CreateSessionResponseMock200({ id: SESSION_ID }),
    ),
    getGetV2GetSessionMockHandler200(
      getGetV2GetSessionResponseMock200({
        id: SESSION_ID,
        messages: [],
        active_stream: null,
      }),
    ),
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${SESSION_ID}/stream`,
      ({ request }) =>
        streamSseResponse(ASK_WHICH_MEMORIES, { abortSignal: request.signal }),
    ),
  );
});

describe("Forget a topic chat", () => {
  it("renders Otto's question card so the user can pick what to forget", async () => {
    render(<SettingsMemoryPage />);

    await screen.findByText("Runs a DTC candle brand called Emberline");
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Forget a topic…" }));

    const panel = await screen.findByRole("complementary", {
      name: "Memory chat panel",
    });
    const options = await within(panel).findByRole("radiogroup");
    expect(
      within(options).getByRole("radio", { name: "Runs Emberline" }),
    ).toBeDefined();
    expect(
      within(options).getByRole("radio", { name: "Monday summary emails" }),
    ).toBeDefined();
    expect(
      within(panel).getByText("Which memories should I forget?"),
    ).toBeDefined();
    expect(
      within(panel).getByRole("button", { name: "Send answers" }),
    ).toBeDefined();
  });
});
