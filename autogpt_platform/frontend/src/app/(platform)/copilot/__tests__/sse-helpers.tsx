import {
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2GetSessionMockHandler200,
  getPostV2CancelSessionTaskMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { streamSseResponse } from "@/tests/integrations/copilot-sse";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http, type HttpHandler } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode, useState } from "react";
import { expect } from "vitest";
import { CopilotChatHost } from "../CopilotChatHost";

export const TEST_BACKEND_BASE_URL = "http://localhost:18006";
export const TEST_SESSION_ID = "test-session-stream-1";

export interface SessionOverride {
  active_stream?: SessionDetailResponse["active_stream"];
  messages?: SessionDetailResponse["messages"];
  has_more_messages?: boolean;
  chat_status?: string;
}

/**
 * Override the default Orval session-GET handler with a clean, fully-controlled
 * session — empty messages, no active stream — so each test starts from a
 * deterministic baseline. Random faker fields are replaced with stable values.
 */
export function sessionHandler(opts: SessionOverride = {}) {
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

/**
 * Multi-call POST handler for the CoPilot stream URL. Each request consumes
 * the next entry in `chunksPerTurn`; after the array is exhausted the last
 * entry is replayed for every subsequent request. Lets a single test exercise
 * back-to-back turns in the same session.
 */
export function copilotStreamSequenceHandler({
  baseUrl,
  sessionId,
  chunksPerTurn,
}: {
  baseUrl: string;
  sessionId: string;
  chunksPerTurn: UIMessageChunk[][];
}): HttpHandler {
  let turn = 0;
  return http.post(
    `${baseUrl}/api/chat/sessions/${sessionId}/stream`,
    ({ request }) => {
      const chunks =
        chunksPerTurn[turn] ?? chunksPerTurn[chunksPerTurn.length - 1];
      turn += 1;
      return streamSseResponse(chunks, { abortSignal: request.signal });
    },
  );
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

/**
 * Render `<CopilotChatHost />` with a pre-set `sessionId` in URL state and
 * neutral overrides for the chat-adjacent endpoints (session, usage, pending
 * messages). The default Orval mocks return random faker data that flips the
 * input into "limit reached" or injects ghost queued chips, so we pin them.
 */
export function renderHost(opts: { sessionOverride?: SessionOverride } = {}) {
  server.use(
    sessionHandler(opts.sessionOverride),
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
    // useCopilotStop POSTs here when the user clicks Stop. The default
    // Orval handler returns random faker fields; pin to a deterministic
    // success so the stop path neither triggers the destructive "Could
    // not stop the task" toast nor the "Stop may take a moment" toast
    // (which would render extra DOM and risk flaky assertions).
    getPostV2CancelSessionTaskMockHandler200({
      cancelled: true,
      reason: null,
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

/**
 * The textarea is disabled while the session GET is loading. Block until it's
 * enabled before driving userEvent — otherwise keystrokes are silently
 * swallowed and the test times out on findByText with no useful trace.
 */
export async function waitForInputReady() {
  await waitFor(() => {
    const input = screen.getByLabelText(
      /chat message input/i,
    ) as HTMLTextAreaElement;
    expect(input.disabled).toBe(false);
  });
}

export async function typeAndSend(text: string) {
  await waitForInputReady();
  const user = userEvent.setup();
  const input = screen.getByLabelText(/chat message input/i);
  await user.type(input, text);
  await user.click(screen.getByRole("button", { name: /submit/i }));
}

export async function clickStop() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: /stop/i }));
}
