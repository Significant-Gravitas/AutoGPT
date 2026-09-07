import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { beforeEach, expect, test, vi } from "vitest";
import {
  getGetV2GetSessionMockHandler200,
  getPostV2CreateSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { useExpertChatDrawer } from "../useExpertChatDrawer";

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  setMessages: vi.fn(),
  stream: vi.fn(),
}));

vi.mock("@/app/(platform)/copilot/useCopilotStream", () => ({
  useCopilotStream: (options: unknown) => {
    mocks.stream(options);
    return {
      messages: [],
      setMessages: mocks.setMessages,
      sendMessage: mocks.sendMessage,
      stop: vi.fn(),
      status: "ready",
      error: undefined,
    };
  },
}));
vi.mock("@/app/(platform)/copilot/useCopilotPendingChips", () => ({
  useCopilotPendingChips: () => ({ queuedMessages: [], queueMessage: vi.fn() }),
}));

beforeEach(() => vi.clearAllMocks());

function wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      {children}
    </QueryClientProvider>
  );
}

test("creates and loads an expert chat in its own tenant and streams there", async () => {
  useOrgTeamStore.setState({
    activeOrgID: "org-nav",
    activeTeamID: "team-nav",
    isLoaded: true,
  });
  const requests: Headers[] = [];
  const session = {
    id: "session-expert",
    created_at: "2026-09-07T00:00:00Z",
    user_id: "user",
    organization_id: "org-expert",
    team_id: null,
  };
  server.use(
    getPostV2CreateSessionMockHandler200(async ({ request }) => {
      requests.push(request.headers);
      expect(await request.json()).toEqual({ expert_id: "expert-one" });
      return session;
    }),
    getGetV2GetSessionMockHandler200(({ request }) => {
      requests.push(request.headers);
      return { ...session, updated_at: session.created_at, messages: [] };
    }),
  );
  const { result } = renderHook(
    () =>
      useExpertChatDrawer({
        target: {
          expertId: "expert-one",
          name: "One",
          role: "Helper",
          avatarUrl: null,
          organizationId: "org-expert",
          teamId: null,
        },
        isOpen: true,
        resumeLatest: false,
        threadKey: 0,
        seedPrompt: null,
      }),
    { wrapper },
  );
  await act(() => result.current.onSend("Hello"));
  await waitFor(() => expect(requests).toHaveLength(2));
  for (const headers of requests) {
    expect(headers.get("X-Org-Id")).toBe("org-expert");
    expect(headers.get("X-Team-Id")).toBeNull();
  }
  expect(mocks.stream).toHaveBeenLastCalledWith(
    expect.objectContaining({
      sessionId: "session-expert",
      sessionTenantScope: { organizationId: "org-expert", teamId: null },
    }),
  );
  expect(mocks.sendMessage).toHaveBeenCalledWith({ text: "Hello" });
});

test("does not create an unscoped Autopilot chat while tenant context loads", async () => {
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    isLoaded: false,
  });
  const create = vi.fn();
  server.use(
    getPostV2CreateSessionMockHandler200(() => {
      create();
      return {
        id: "unexpected",
        created_at: "2026-09-07T00:00:00Z",
        user_id: "user",
      };
    }),
  );
  const { result } = renderHook(
    () =>
      useExpertChatDrawer({
        target: {
          expertId: null,
          name: "Autopilot",
          role: "Helper",
          avatarUrl: null,
        },
        isOpen: true,
        resumeLatest: false,
        threadKey: 0,
        seedPrompt: null,
      }),
    { wrapper },
  );
  await act(() => result.current.onSend("Hello"));
  expect(result.current.isScopeReady).toBe(false);
  expect(create).not.toHaveBeenCalled();
});
