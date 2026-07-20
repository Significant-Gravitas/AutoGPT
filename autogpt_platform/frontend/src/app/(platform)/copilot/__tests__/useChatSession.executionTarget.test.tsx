import type { NewChatExecutionTarget } from "../store";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const SESSIONS_URL = "http://localhost:3000/api/proxy/api/chat/sessions";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
  getTraceData: vi.fn(() => ({})),
}));

function renderChatSession(target?: NewChatExecutionTarget) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  function TestProviders({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <NuqsTestingAdapter>{children}</NuqsTestingAdapter>
      </QueryClientProvider>
    );
  }

  return renderHook(() => useChatSession({ executionTarget: target }), {
    wrapper: TestProviders,
  });
}

function createdSession(executionTarget: { kind: string } | null = null) {
  return {
    id: "session-1",
    created_at: "2026-07-10T17:00:00Z",
    user_id: "user-1",
    metadata: executionTarget
      ? { dry_run: false, execution_target: executionTarget }
      : { dry_run: false },
  };
}

function mockCreatedSession(executionTarget: { kind: string } | null = null) {
  server.use(
    http.get(`${SESSIONS_URL}/session-1`, () =>
      HttpResponse.json({
        ...createdSession(executionTarget),
        updated_at: "2026-07-10T17:00:00Z",
        chat_status: "idle",
        messages: [],
        active_stream: null,
        has_more_messages: false,
        oldest_sequence: null,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
      }),
    ),
  );
}

afterEach(() => {
  useCopilotUIStore.setState({
    newChatExecutionTarget: { kind: "cloud" },
    isExecutionTargetPickerOpen: false,
    executionTargetError: null,
  });
});

describe("useChatSession execution target", () => {
  it("creates a Cloud session explicitly when the picker is enabled", async () => {
    let requestBody: unknown;
    mockCreatedSession({ kind: "cloud" });
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestBody = await request.json();
        return HttpResponse.json(createdSession({ kind: "cloud" }));
      }),
    );

    const session = renderChatSession({ kind: "cloud" });
    await act(async () => {
      await session.result.current.createSession();
    });

    await waitFor(() => {
      expect(requestBody).toEqual({ execution_target: { kind: "cloud" } });
    });
  });

  it("sends the exact selected machine and opaque folder references", async () => {
    let requestBody: unknown;
    mockCreatedSession({ kind: "local" });
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestBody = await request.json();
        return HttpResponse.json(createdSession({ kind: "local" }));
      }),
    );

    const session = renderChatSession({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: "directory-1",
      displayPath: "C:\\Users\\Ada\\Projects",
    });
    await act(async () => {
      await session.result.current.createSession();
    });

    await waitFor(() => {
      expect(requestBody).toEqual({
        execution_target: {
          kind: "local",
          machine_id: "machine-1",
          expected_connection_id: "connection-1",
          browse_id: "browse-1",
          directory_ref: "directory-1",
        },
      });
    });
  });

  it("fails closed on stale Local PC selection without retrying in Cloud", async () => {
    const requestBodies: unknown[] = [];
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestBodies.push(await request.json());
        return HttpResponse.json(
          { detail: "Executor connection changed" },
          { status: 409 },
        );
      }),
    );

    const session = renderChatSession({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "stale-connection",
      browseID: "browse-1",
      directoryRef: "directory-1",
      displayPath: "C:\\Projects",
    });
    await act(async () => {
      await expect(session.result.current.createSession()).rejects.toThrow(
        /connection changed/i,
      );
    });

    await waitFor(() => {
      expect(useCopilotUIStore.getState().isExecutionTargetPickerOpen).toBe(
        true,
      );
    });
    expect(useCopilotUIStore.getState().executionTargetError).toMatch(
      /changed/i,
    );
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: null,
      browseID: null,
      directoryRef: null,
      displayPath: null,
    });
    expect(requestBodies).toHaveLength(1);
    expect(requestBodies[0]).toMatchObject({
      execution_target: { kind: "local" },
    });
  });

  it("keeps the feature-off session request free of execution metadata", async () => {
    let requestText: string | undefined;
    mockCreatedSession();
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestText = await request.text();
        return HttpResponse.json(createdSession());
      }),
    );

    const session = renderChatSession();
    await act(async () => {
      await session.result.current.createSession();
    });

    await waitFor(() => expect(requestText).toBe("null"));
  });
});
