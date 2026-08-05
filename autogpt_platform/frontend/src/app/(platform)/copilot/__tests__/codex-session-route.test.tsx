import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";

const testState = vi.hoisted(() => ({
  transports: [] as ChatTransportResponse[] | null,
  transportError: false,
  queryOptions: vi.fn(),
  toast: vi.fn(),
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
      useGetV2ListChatTransports: (options: unknown) => {
        testState.queryOptions(options);
        return {
          data:
            testState.transports === null
              ? undefined
              : { status: 200, data: { transports: testState.transports } },
          isError: testState.transportError,
        };
      },
    };
  },
);

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => testState.toast(...args),
  useToast: () => ({ toast: testState.toast, dismiss: vi.fn() }),
}));

const hostedPlatform: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "AutoGPT Platform",
  available: true,
  default: true,
};

const configuredSelfHosted: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "Self-hosted chat",
  available: true,
  default: true,
};

const unconfiguredSelfHosted: ChatTransportResponse = {
  ...configuredSelfHosted,
  available: false,
  default: false,
};

const codexTransport: ChatTransportResponse = {
  auth_provider: "codex",
  credential_id: "codex-credential-1",
  label: "ChatGPT/Codex",
  available: true,
  default: false,
};

function SessionHarness() {
  const { createSession, setSessionId, sessionLlmAuthProvider } =
    useChatSession();
  return (
    <>
      <div data-testid="session-llm-route">
        {sessionLlmAuthProvider ?? "unresolved"}
      </div>
      <button
        type="button"
        onClick={() => void createSession().catch(() => {})}
      >
        Create session
      </button>
      <button type="button" onClick={() => void setSessionId("existing")}>
        Open existing session
      </button>
    </>
  );
}

function captureCreateRequest() {
  let requestBody: unknown;
  server.use(
    http.post("*/api/chat/sessions", async ({ request }) => {
      requestBody = await request.json();
      return HttpResponse.json({
        id: "new-session-1",
        created_at: "2026-01-01T00:00:00Z",
        user_id: "user-1",
      });
    }),
  );
  return () => requestBody;
}

afterEach(() => {
  cleanup();
  server.resetHandlers();
  testState.transports = [];
  testState.transportError = false;
  testState.queryOptions.mockClear();
  testState.toast.mockClear();
  useCopilotUIStore.getState().setCopilotLlmAuth({
    authProvider: "platform",
    credentialId: null,
  });
});

describe("useChatSession transport route", () => {
  it("does not keep a fresh transport inventory after focus changes", () => {
    testState.transports = [hostedPlatform];

    render(<SessionHarness />);

    expect(testState.queryOptions).toHaveBeenLastCalledWith(
      expect.objectContaining({
        query: expect.objectContaining({
          enabled: true,
          refetchOnWindowFocus: true,
          staleTime: 0,
        }),
      }),
    );
  });

  it("sends the hosted platform default when Codex is also available", async () => {
    testState.transports = [hostedPlatform, codexTransport];
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({ llm_auth_provider: "platform" });
    });
  });

  it("sends the explicitly selected hosted Codex credential", async () => {
    testState.transports = [hostedPlatform, codexTransport];
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({
        llm_auth_provider: "codex",
        llm_credential_id: "codex-credential-1",
      });
    });
  });

  it("sends configured self-hosted chat as the sole route", async () => {
    testState.transports = [configuredSelfHosted];
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({ llm_auth_provider: "platform" });
    });
  });

  it("selects Codex automatically on a keyless self-host", async () => {
    testState.transports = [
      unconfiguredSelfHosted,
      { ...codexTransport, default: true },
    ];
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({
        llm_auth_provider: "codex",
        llm_credential_id: "codex-credential-1",
      });
    });
  });

  it("blocks a keyless self-host with no Codex before session creation", async () => {
    testState.transports = [unconfiguredSelfHosted];
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          title: "AutoPilot needs an AI connection",
        }),
      );
    });
    expect(getRequestBody()).toBeUndefined();
  });

  it("blocks while the transport inventory is loading", async () => {
    testState.transports = null;
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "AI connections are still loading",
        }),
      );
    });
    expect(getRequestBody()).toBeUndefined();
  });

  it("blocks when the transport inventory fails", async () => {
    testState.transports = null;
    testState.transportError = true;
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not check AI connections",
        }),
      );
    });
    expect(getRequestBody()).toBeUndefined();
  });

  it("requires a choice when a keyless self-host has multiple Codex routes", async () => {
    testState.transports = [
      unconfiguredSelfHosted,
      codexTransport,
      { ...codexTransport, credential_id: "codex-credential-2" },
    ];
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Choose an AI connection" }),
      );
    });
    expect(getRequestBody()).toBeUndefined();
  });

  it("restores the immutable route from existing session metadata", async () => {
    server.use(
      http.get("*/api/chat/sessions/existing", () =>
        HttpResponse.json({
          id: "existing",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          metadata: { llm_auth_provider: "codex" },
          messages: [],
        }),
      ),
    );
    render(<SessionHarness />);

    fireEvent.click(
      screen.getByRole("button", { name: "Open existing session" }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("session-llm-route").textContent).toBe("codex");
    });
  });
});
