import {
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2GetSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import {
  assistantTextChunks,
  copilotResumeHandler,
  copilotStreamErrorHandler,
  copilotStreamHandler,
} from "@/tests/integrations/copilot-sse";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CopilotChatHost } from "../CopilotChatHost";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";

const TEST_BACKEND_BASE_URL = "http://localhost:18006";
const TEST_SESSION_ID = "test-session-stream-1";

// Pin the backend host so the CoPilot transport's absolute URL is
// deterministic — the transport bypasses the Next proxy on purpose
// (Vercel function-timeout dodge), so MSW has to match an absolute URL.
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

// Replace the Supabase token fetch with a static header so we don't
// need real auth in tests.
vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

// CoPilotPage gates the page on a logged-in Supabase user; CopilotChatHost
// itself doesn't read useSupabase but useCopilotPage does, via useChatSession.
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

// Keep mode/model toggles and artifacts off so the chat input renders a
// single, predictable Submit button.
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ARTIFACTS: "ARTIFACTS",
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
  },
  useGetFlag: () => false,
}));

interface SessionOverride {
  active_stream?: SessionDetailResponse["active_stream"];
  messages?: SessionDetailResponse["messages"];
  has_more_messages?: boolean;
  chat_status?: string;
}

function sessionHandler(opts: SessionOverride = {}) {
  return getGetV2GetSessionMockHandler200({
    id: TEST_SESSION_ID,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    user_id: "test-user",
    chat_status: opts.chat_status ?? "idle",
    messages: opts.messages ?? [],
    has_more_messages: opts.has_more_messages ?? false,
    oldest_sequence: null,
    active_stream: opts.active_stream ?? null,
    metadata: { dry_run: false, builder_graph_id: null },
  });
}

function Wrapper({
  children,
  searchParams,
}: {
  children: ReactNode;
  searchParams: string;
}) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter searchParams={searchParams}>
        <BackendAPIProvider>
          <OnboardingProvider>
            <TooltipProvider>{children}</TooltipProvider>
          </OnboardingProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

function renderHost(opts: { sessionOverride?: SessionOverride } = {}) {
  server.use(
    sessionHandler(opts.sessionOverride),
    // Default Orval mocks return random faker data. The usage handler's
    // random `percent_used` can land >= 100 and lock the input as
    // "limit reached"; the pending-messages handler invents random
    // queued chips that pollute the message list. Pin both to neutral
    // values so the only state in play is what the test sets up.
    getGetV2GetCopilotUsageMockHandler200({
      daily: {
        percent_used: 0,
        resets_at: new Date("2026-05-14T00:00:00Z"),
      },
      weekly: {
        percent_used: 0,
        resets_at: new Date("2026-05-20T00:00:00Z"),
      },
      tier: "PRO",
      reset_cost: 0,
    }),
    getGetV2GetPendingMessagesMockHandler200({
      count: 0,
      messages: [],
    }),
  );
  return render(
    <CopilotChatHost droppedFiles={[]} onDroppedFilesConsumed={() => {}} />,
    {
      wrapper: ({ children }) => (
        <Wrapper searchParams={`?sessionId=${TEST_SESSION_ID}`}>
          {children}
        </Wrapper>
      ),
    },
  );
}

async function waitForInputReady() {
  // The textarea is disabled while the session GET is loading. Block until
  // it's enabled before driving userEvent, otherwise typing is silently
  // swallowed and tests time out on findByText with no useful trace.
  await waitFor(() => {
    const input = screen.getByLabelText(
      /chat message input/i,
    ) as HTMLTextAreaElement;
    expect(input.disabled).toBe(false);
  });
}

async function typeAndSend(text: string) {
  await waitForInputReady();
  const user = userEvent.setup();
  const input = screen.getByLabelText(/chat message input/i);
  await user.type(input, text);
  await user.click(screen.getByRole("button", { name: /submit/i }));
}

async function clickStop() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: /stop/i }));
}

beforeEach(() => {
  resetCopilotChatRegistry();
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("CoPilot streaming (SSE) — content rendering", () => {
  it("renders assistant text from a single text-delta frame", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hello from the copilot."),
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Hello from the copilot.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("concatenates multiple text-delta frames into the final assistant message", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Hello " },
          { type: "text-delta", id: "t1", delta: "from " },
          { type: "text-delta", id: "t1", delta: "the copilot." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Hello from the copilot.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("renders the assistant's final text after reasoning chunks", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "reasoning-start", id: "r1" },
          { type: "reasoning-delta", id: "r1", delta: "Thinking " },
          { type: "reasoning-delta", id: "r1", delta: "step by step." },
          { type: "reasoning-end", id: "r1" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Final answer." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText("Final answer.", undefined, { timeout: 5000 }),
    ).toBeDefined();
  });

  it("renders text emitted after a tool call across the same turn", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          {
            type: "tool-input-start",
            toolCallId: "call-1",
            toolName: "search",
            dynamic: true,
          },
          {
            type: "tool-input-available",
            toolCallId: "call-1",
            toolName: "search",
            input: { query: "weather" },
            dynamic: true,
          },
          {
            type: "tool-output-available",
            toolCallId: "call-1",
            output: { result: "sunny" },
            dynamic: true,
          },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "The weather is sunny." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("weather?");

    expect(
      await screen.findByText("The weather is sunny.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("renders text from both steps of a two-step turn", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "First step text." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "start-step" },
          { type: "text-start", id: "t2" },
          { type: "text-delta", id: "t2", delta: "Second step text." },
          { type: "text-end", id: "t2" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    // Each step's text part renders in its own element, so assert both
    // individually rather than as a single concatenated string.
    expect(
      await screen.findByText("First step text.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(await screen.findByText("Second step text.")).toBeDefined();
  });

  it("completes the turn cleanly on an empty completion (no content, no error)", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    await waitFor(
      () => {
        expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
        expect(
          screen.queryByRole("button", { name: /submit/i }),
        ).not.toBeNull();
      },
      { timeout: 5000 },
    );
    expect(screen.queryByText(/encountered an error/i)).toBeNull();
  });
});

describe("CoPilot streaming (SSE) — status lifecycle", () => {
  it("swaps the submit button to Stop while streaming and back to Submit when done", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hi."),
        delayMsBetweenChunks: 30,
      }),
    );

    renderHost();
    expect(screen.getByRole("button", { name: /submit/i })).toBeDefined();

    await typeAndSend("hi");

    expect(await screen.findByRole("button", { name: /stop/i })).toBeDefined();

    expect(
      await screen.findByText("Hi.", undefined, { timeout: 5000 }),
    ).toBeDefined();
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
    });
  });
});

describe("CoPilot streaming (SSE) — error paths", () => {
  it("surfaces an SSE error chunk to the user", async () => {
    const chunks: UIMessageChunk[] = [
      { type: "start", messageId: "msg-1" },
      { type: "start-step" },
      { type: "error", errorText: "Backend went sideways." },
    ];
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks,
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText(/backend went sideways\./i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("opens the rate-limit dialog on HTTP 429 'usage limit'", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 429,
        body: { detail: "You've reached your usage limit. Try again later." },
      }),
    );

    renderHost();
    await typeAndSend("rate limited please");

    // useCopilotStream's rate-limit branch sets rateLimitMessage, which the
    // RateLimitGate translates into a Dialog with this title.
    expect(
      await screen.findByText(/daily autopilot limit reached/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });

  it("surfaces an HTTP 500 response as a visible error", async () => {
    server.use(
      copilotStreamErrorHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        status: 500,
        body: "kaboom",
      }),
    );

    renderHost();
    await typeAndSend("hi");

    expect(
      await screen.findByText(/kaboom/i, undefined, { timeout: 5000 }),
    ).toBeDefined();
  });
});

describe("CoPilot streaming (SSE) — stop", () => {
  it("ends the stream and shows the manual-stop marker when Stop is clicked mid-stream", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: [
          { type: "start", messageId: "msg-1" },
          { type: "start-step" },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Streaming " },
          { type: "text-delta", id: "t1", delta: "in progress " },
          { type: "text-delta", id: "t1", delta: "should be cut off" },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
        // Setup + first delta emit instantly; stall before the "should be cut
        // off" delta so the stop click has a wide window to land.
        perChunkDelaysMs: [0, 0, 0, 0, 5000, 0, 0, 0, 0],
      }),
    );

    renderHost();
    await typeAndSend("hi");

    // Wait for the first delta so Stop has something live to abort.
    await screen.findByText(/streaming/i, undefined, { timeout: 5000 });

    await clickStop();

    expect(
      await screen.findByText(/you manually stopped this chat/i, undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    // Post-stop chunks never reach the bubble.
    expect(screen.queryByText(/should be cut off/i)).toBeNull();
  });
});

describe("CoPilot streaming (SSE) — resume on mount", () => {
  it("issues a GET resume and renders streamed content when the session has an active_stream", async () => {
    server.use(
      copilotResumeHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Resumed content here."),
      }),
    );

    renderHost({
      sessionOverride: {
        active_stream: {
          turn_id: "turn-1",
          last_message_id: "msg-prev",
          started_at: "2026-05-13T00:00:00Z",
        },
      },
    });

    expect(
      await screen.findByText("Resumed content here.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
  });
});
