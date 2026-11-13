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

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureException: vi.fn(),
  captureMessage: vi.fn(),
  getTraceData: vi.fn(() => ({})),
}));

vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/chat/chat")
      >();
    return {
      ...actual,
      useGetV2ListChatTransports: () => ({
        data: {
          status: 200,
          data: {
            transports: [
              {
                auth_provider: "platform",
                credential_id: null,
                label: "AutoGPT Platform",
                available: true,
                default: true,
              },
              {
                auth_provider: "codex",
                credential_id: "credential-1",
                label: "ChatGPT",
                available: true,
                default: false,
              },
            ],
          },
        },
        isError: false,
      }),
    };
  },
);

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

function mockCreatedSession(
  executionTarget: { kind: string; [field: string]: unknown } | null = null,
) {
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
    copilotLlmAuth: null,
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

    // Every new chat names itself so a retry adopts it (dev's create flow).
    await waitFor(() => {
      expect(requestBody).toEqual({
        session_id: expect.any(String),
        execution_target: { kind: "cloud" },
      });
    });
  });

  it.each([false, true])(
    "sends the selected machine and folder with explicit ChatGPT connection=%s",
    async (useCodex) => {
      if (useCodex) {
        useCopilotUIStore.getState().setCopilotLlmAuth({
          authProvider: "codex",
          credentialId: "credential-1",
        });
      }
      let requestBody: unknown;
      mockCreatedSession({
        kind: "local",
        machine_id: "machine-1",
        allowed_root: "C:\\Users\\Ada\\Projects",
      });
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
          ...(useCodex
            ? { llm_auth_provider: "codex", llm_credential_id: "credential-1" }
            : {}),
          session_id: expect.any(String),
          execution_target: {
            kind: "local",
            machine_id: "machine-1",
            expected_connection_id: "connection-1",
            browse_id: "browse-1",
            directory_ref: "directory-1",
          },
        });
      });
      // The created session reads back as bound to that machine and folder.
      // That takes the create and then the session read, so a loaded runner
      // gets more than Testing Library's 3 s default.
      await waitFor(
        () => {
          expect(session.result.current.sessionExecutionTarget).toEqual({
            kind: "local",
            machine_id: "machine-1",
            allowed_root: "C:\\Users\\Ada\\Projects",
          });
        },
        { timeout: 10_000 },
      );
    },
  );

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
        /computer or folder changed/i,
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

  it("blocks an incomplete Local PC selection before sending a create request", async () => {
    const requestBodies: unknown[] = [];
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestBodies.push(await request.json());
        return HttpResponse.json(createdSession());
      }),
    );

    const session = renderChatSession({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: null,
      displayPath: null,
    });
    await act(async () => {
      await expect(session.result.current.createSession()).rejects.toThrow(
        /Choose a connected computer and folder/i,
      );
    });

    expect(requestBodies).toHaveLength(0);
    expect(useCopilotUIStore.getState().isExecutionTargetPickerOpen).toBe(true);
  });

  it("keeps the chosen folder when only the session id was taken", async () => {
    server.use(
      http.post(SESSIONS_URL, () =>
        HttpResponse.json(
          { detail: "Session id already in use" },
          { status: 409 },
        ),
      ),
    );
    const selected = {
      kind: "local" as const,
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: "directory-1",
      displayPath: "C:\\Projects",
    };

    const session = renderChatSession(selected);
    useCopilotUIStore.getState().setNewChatExecutionTarget(selected);
    const before = useCopilotUIStore.getState().newChatExecutionTarget;
    expect(before).toEqual(selected);
    await act(async () => {
      await expect(session.result.current.createSession()).rejects.toThrow();
    });

    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual(before);
    expect(useCopilotUIStore.getState().executionTargetError).toBeNull();
    expect(useCopilotUIStore.getState().isExecutionTargetPickerOpen).toBe(
      false,
    );
  });

  it("keeps the feature-off session request free of execution metadata", async () => {
    let requestBody: unknown;
    mockCreatedSession();
    server.use(
      http.post(SESSIONS_URL, async ({ request }) => {
        requestBody = await request.json();
        return HttpResponse.json(createdSession());
      }),
    );

    const session = renderChatSession();
    await act(async () => {
      await session.result.current.createSession();
    });

    await waitFor(() =>
      expect(requestBody).toEqual({ session_id: expect.any(String) }),
    );
  });
});
