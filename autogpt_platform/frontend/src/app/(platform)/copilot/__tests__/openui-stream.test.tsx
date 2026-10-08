import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { http } from "msw";
import type { UIMessageChunk } from "ai";
import { server } from "@/mocks/mock-server";
import { campaign } from "@/lib/openui/samples";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  renderHost,
  resetChatRuntimes,
  STREAM_PATHS,
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
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));
const streamPath = vi.hoisted(() => ({ runtime: false }));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    COPILOT_STREAM_RUNTIME: "copilot-stream-runtime",
  },
  useGetFlag: (flag: string) =>
    flag === "copilot-stream-runtime" ? streamPath.runtime : false,
}));

beforeEach(() => {
  resetChatRuntimes();
  sessionStorage.clear();
});
afterEach(() => {
  resetChatRuntimes();
});

const chunks: UIMessageChunk[] = [
  { type: "start", messageId: "ui-msg" },
  { type: "start-step" },
  {
    type: "tool-input-start",
    toolCallId: "ui-call",
    toolName: "render_ui",
    dynamic: true,
  },
  {
    type: "tool-input-available",
    toolCallId: "ui-call",
    toolName: "render_ui",
    dynamic: true,
    input: { source: campaign, summary: "Editable campaign brief." },
  },
  {
    type: "tool-output-available",
    toolCallId: "ui-call",
    dynamic: true,
    output: JSON.stringify({
      type: "ui_rendered",
      version: 1,
      source: campaign,
      message: "Editable campaign brief.",
    }),
  },
  { type: "finish-step" },
  { type: "finish" },
];

describe.each(STREAM_PATHS)("OpenUI in the real Copilot host on %s", (path) => {
  it("renders a streamed tool result and sends edited form values through the same conversation", async () => {
    streamPath.runtime = path === "stream runtime";
    const requests: string[] = [];
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async ({ request }) => {
          requests.push(await request.text());
          return streamSseResponse(
            requests.length === 1
              ? chunks
              : assistantTextChunks(
                  "The campaign plan now targets independent bookshops.",
                  { messageId: "follow-up-msg" },
                ),
            { abortSignal: request.signal },
          );
        },
      ),
    );
    renderHost();
    await typeAndSend("Help me plan a campaign with an editable brief.");
    const audience = await screen.findByRole(
      "textbox",
      { name: "Audience" },
      { timeout: 10000 },
    );
    await waitFor(() => expect(audience.hasAttribute("disabled")).toBe(false));
    fireEvent.change(audience, { target: { value: "Independent bookshops" } });
    fireEvent.click(screen.getByRole("button", { name: "Build my plan" }));
    expect(
      await screen.findByText(
        "The campaign plan now targets independent bookshops.",
        undefined,
        { timeout: 10000 },
      ),
    ).toBeDefined();
    expect(requests).toHaveLength(2);
    expect(requests[1]).toContain("Independent bookshops");
    expect(requests[1]).toContain("Submitted values");
  }, 20_000);
});
