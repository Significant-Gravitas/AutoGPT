import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import {
  getGetV2GetSessionMockHandler200,
  getGetV2GetSessionResponseMock200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetMyExpertMemoryOverviewMockHandler200,
  getGetMyMemoryOverviewMockHandler200,
  getListMyExpertMemoryFactsMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { useIsMutating } from "@tanstack/react-query";
import { http } from "msw";
import { useState } from "react";
import { expect, vi } from "vitest";
import SettingsMemoryPage from "../page";

const mocks = vi.hoisted(() => ({
  captureException: vi.fn(),
  config: { billingEnabled: true },
}));
export const captureException = mocks.captureException;
export const memoryChatTestConfig = mocks.config;

vi.mock("@sentry/nextjs", () => ({
  captureMessage: vi.fn(),
  captureException: mocks.captureException,
  addBreadcrumb: vi.fn(),
  getTraceData: () => ({}),
}));

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      isCloud: () => false,
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

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.ENABLE_PLATFORM_PAYMENT &&
      mocks.config.billingEnabled,
  };
});

function MemoryPageHarness() {
  const [visible, setVisible] = useState(true);
  const pending = useIsMutating({ mutationKey: ["postV2CreateSession"] });
  return (
    <>
      {visible ? <SettingsMemoryPage /> : null}
      <Toaster />
      <span data-testid="pending-session-requests">{pending}</span>
      <button onClick={() => setVisible(false)}>Leave memory settings</button>
    </>
  );
}

export function renderMemoryPage() {
  return render(<MemoryPageHarness />);
}

export async function waitForSessionCreates(count: number) {
  await waitFor(() =>
    expect(screen.getByTestId("pending-session-requests").textContent).toBe(
      String(count),
    ),
  );
}

export function setupMemoryChat() {
  resetCopilotChatRegistry();
  captureException.mockClear();
  memoryChatTestConfig.billingEnabled = true;
  window.history.replaceState({}, "", "/settings/memory");
  const sessionID = "memory-paywall-test-session";
  server.use(
    getListExpertsMockHandler200([
      {
        ...getListExpertsResponseMock200()[0],
        id: "expert-maria",
        name: "Maria",
        role: "Growth Marketer",
        avatar_url: null,
        is_archived: false,
      },
    ]),
    getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
    getListMyExpertMemoryFactsMockHandler200({
      expert_id: "expert-maria",
      items: [],
    }),
    getGetMyExpertMemoryOverviewMockHandler200({
      expert_id: "expert-maria",
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
    getPostV2CreateSessionMockHandler200(
      getPostV2CreateSessionResponseMock200({ id: sessionID }),
    ),
    getGetV2GetSessionMockHandler200(
      getGetV2GetSessionResponseMock200({
        id: sessionID,
        messages: [],
        active_stream: null,
      }),
    ),
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${sessionID}/stream`,
      ({ request }) =>
        streamSseResponse(assistantTextChunks("Your memory summary."), {
          abortSignal: request.signal,
        }),
    ),
  );
}
