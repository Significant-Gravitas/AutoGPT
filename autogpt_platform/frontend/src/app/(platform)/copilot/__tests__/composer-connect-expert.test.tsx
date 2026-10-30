import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ChatContainer } from "../components/ChatContainer/ChatContainer";
import { CopilotModals } from "../components/CopilotModals/CopilotModals";
import { useCopilotUIStore } from "../store";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("../components/ChatMessagesContainer/ChatMessagesContainer", () => ({
  ChatMessagesContainer: () => <div data-testid="chat-messages-container" />,
}));

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

const maria = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Marketing Strategist",
  isArchived: false,
  readOnlyReason: null,
};

const baseProps = {
  messages: [] as never[],
  status: "ready",
  error: undefined,
  isLoadingSession: false,
  isSessionError: false,
  isCreatingSession: false,
  isReconnecting: false,
  isRestoringActiveSession: false,
  restoreStatusMessage: null,
  activeStreamStartedAt: null,
  isUserStopping: false,
  isSyncing: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
  onStop: vi.fn(),
  onEnqueue: vi.fn(),
  queuedMessages: [],
  isUploadingFiles: false,
  hasMoreMessages: false,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  droppedFiles: [],
  onDroppedFilesConsumed: vi.fn(),
};

function arrangeSentry() {
  const requests = { token: 0, grants: 0 };
  server.use(
    getGetV1ListCredentialsMockHandler([]),
    getGetV1ListProvidersMockHandler([
      {
        name: "mcp_sentry",
        display_name: "Sentry",
        description: "Errors",
        supported_auth_types: [],
        service: "sentry",
        service_name: "Sentry",
        service_icon: "sentry",
        mcp_server: {
          server_url: "https://mcp.sentry.dev/mcp",
          documentation_url: "https://docs.sentry.io",
          setup_instructions: "Paste your Sentry token.",
          connection_mode: "hosted",
          auth_methods: ["bearer"],
          oauth_write_scopes: [],
          server_url_options: [],
          allow_custom_url: false,
        },
      },
    ]),
    http.post("*/api/mcp/discover-tools", () =>
      HttpResponse.json({
        tools: [],
        server_url: "https://mcp.sentry.dev/mcp",
      }),
    ),
    http.post("*/api/mcp/token", () => {
      requests.token += 1;
      return HttpResponse.json({
        id: "cred-sentry",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.sentry.dev",
        service: "sentry",
      });
    }),
    http.post("*/api/experts/:expertId/credentials", () => {
      requests.grants += 1;
      return HttpResponse.json([]);
    }),
  );
  return requests;
}

beforeEach(() => {
  useCopilotUIStore.setState({ contextPanelExpert: null });
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
});

afterEach(() => {
  useCopilotUIStore.setState({ contextPanelExpert: null });
  vi.unstubAllGlobals();
});

describe("connecting a service from the composer", () => {
  test("a new chat after an expert chat grants nothing to that expert", async () => {
    const requests = arrangeSentry();
    const { rerender } = render(
      <>
        <ChatContainer
          key="chat-host-session-maria"
          {...baseProps}
          sessionId="session-maria"
          expertIdentity={maria}
        />
        <CopilotModals />
      </>,
    );
    await waitFor(() =>
      expect(useCopilotUIStore.getState().contextPanelExpert).toEqual({
        id: "expert-maria",
        name: "Maria",
      }),
    );

    rerender(
      <>
        <ChatContainer
          key="chat-host-new"
          {...baseProps}
          sessionId={null}
          expertIdentity={null}
        />
        <CopilotModals />
      </>,
    );

    await userEvent.click(
      await screen.findByRole("button", { name: "Add files and more" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Connect service/ }),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /Sentry/ }),
    );
    fireEvent.change(await within(dialog).findByLabelText("API token"), {
      target: { value: "sntrys_token" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save token" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(requests.token).toBe(1);
    expect(requests.grants).toBe(0);
    expect(useCopilotUIStore.getState().contextPanelExpert).toBeNull();
  });

  test("an expert resolving from cache cannot receive a new grant", async () => {
    const requests = arrangeSentry();
    render(
      <>
        <ChatContainer
          {...baseProps}
          sessionId={null}
          expertIdentity={maria}
          isResolvingExpertIdentity
        />
        <CopilotModals />
      </>,
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().contextPanelExpert).toBeNull(),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Add files and more" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Connect service/ }),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /Sentry/ }),
    );
    fireEvent.change(await within(dialog).findByLabelText("API token"), {
      target: { value: "sntrys_token" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save token" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(requests.token).toBe(1);
    expect(requests.grants).toBe(0);
  });
});
