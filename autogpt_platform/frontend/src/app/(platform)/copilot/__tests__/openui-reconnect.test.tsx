import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { http } from "msw";
import { server } from "@/mocks/mock-server";
import { campaign } from "@/lib/openui/__tests__/sample-fixtures";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { getOrCreateCopilotChatRuntime } from "../copilotChatRegistry";
import {
  renderHost,
  resetChatRuntimes,
  sessionHandler,
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
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {},
  useGetFlag: () => false,
}));

beforeEach(resetChatRuntimes);
afterEach(() => {
  vi.useRealTimers();
  resetChatRuntimes();
});

it("replaces a stalled POST with one resume, renders its form, and still accepts the next turn", async () => {
  const url = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
  let postAborted = false;
  let posts = 0;
  let resumes = 0;
  const errors = vi.spyOn(console, "error");
  server.use(
    http.post(url, ({ request }) => {
      posts++;
      request.signal.addEventListener("abort", () => {
        postAborted = true;
      });
      return streamSseResponse(
        assistantTextChunks(posts === 1 ? "Old answer" : "Follow-up received"),
        {
          abortSignal: request.signal,
          perChunkDelaysMs: posts === 1 ? [0, 0, 0, 120_000] : undefined,
        },
      );
    }),
    http.get(url, ({ request }) => {
      resumes++;
      return streamSseResponse(
        [
          { type: "start", messageId: "resumed-ui" },
          { type: "start-step" },
          {
            type: "tool-input-available",
            toolCallId: "render-resumed",
            toolName: "render_ui",
            dynamic: true,
            input: { source: campaign, summary: "Recovered campaign" },
          },
          {
            type: "tool-output-available",
            toolCallId: "render-resumed",
            dynamic: true,
            output: JSON.stringify({
              type: "ui_rendered",
              version: 1,
              source: campaign,
              message: "Recovered campaign",
            }),
          },
          { type: "finish-step" },
          { type: "finish" },
        ],
        { abortSignal: request.signal },
      );
    }),
  );
  renderHost();
  await typeAndSend("Build my campaign");
  const runtime = getOrCreateCopilotChatRuntime(TEST_SESSION_ID);
  await waitFor(() => expect(runtime.chat.status).toBe("streaming"));

  server.use(
    sessionHandler({
      active_stream: { turn_id: "turn-1", last_message_id: "0-0" },
    }),
  );
  const visibility = vi.spyOn(document, "visibilityState", "get");
  vi.useFakeTimers({ toFake: ["Date"] });
  visibility.mockReturnValue("hidden");
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
  vi.setSystemTime(Date.now() + 31_000);
  visibility.mockReturnValue("visible");
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
  vi.useRealTimers();

  await screen.findByRole("textbox", { name: "Audience" }, { timeout: 10000 });
  expect(postAborted).toBe(true);
  expect(posts).toBe(1);
  expect(resumes).toBe(1);
  expect(screen.queryByText("Old answer")).toBeNull();
  server.use(sessionHandler());
  await waitFor(() => expect(runtime.chat.status).toBe("ready"));
  await typeAndSend("Continue in this chat");
  await screen.findByText("Follow-up received", undefined, { timeout: 10000 });
  expect(posts).toBe(2);
  expect(
    errors.mock.calls
      .flat()
      .some((error) => String(error).includes("reading 'state'")),
  ).toBe(false);
}, 25_000);
