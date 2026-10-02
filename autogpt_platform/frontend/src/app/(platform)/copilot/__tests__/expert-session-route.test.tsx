import { getGetExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";

const testState = vi.hoisted(() => ({
  transports: [] as ChatTransportResponse[],
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
      useGetV2ListChatTransports: () => ({
        data: { status: 200, data: { transports: testState.transports } },
        isError: false,
      }),
    };
  },
);

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => testState.toast(...args),
  useToast: () => ({ toast: testState.toast, dismiss: vi.fn() }),
}));

const EXPERT_ID = "expert-1";

const codexOne: ChatTransportResponse = {
  auth_provider: "codex",
  credential_id: "codex-credential-1",
  label: "ChatGPT",
  available: true,
  default: false,
};

const codexTwo: ChatTransportResponse = {
  ...codexOne,
  credential_id: "codex-credential-2",
};

const unconfiguredSelfHosted: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "Self-hosted chat",
  available: false,
  default: false,
};

const pinnedExpert = {
  id: EXPERT_ID,
  name: "Maria",
  avatar_url: null,
  role: "Marketing",
  tagline: null,
  bio: null,
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  llm_auth_provider: "codex",
  llm_credential_id: "codex-credential-2",
  llm_route_label: "ChatGPT",
  llm_route_available: true,
} as unknown as Expert;

const unpinnedExpert: Expert = {
  ...pinnedExpert,
  llm_auth_provider: null,
  llm_credential_id: null,
  llm_route_label: null,
};

function SessionHarness() {
  const { createSession } = useChatSession({
    expertId: EXPERT_ID,
    adoptLatestExpertThread: false,
  });
  return (
    <button type="button" onClick={() => void createSession().catch(() => {})}>
      Create session
    </button>
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
        expert_id: EXPERT_ID,
      });
    }),
  );
  return () => requestBody;
}

async function expertHasLoaded() {
  // The expert query is what tells the hook a pin exists; a click before it
  // settles would exercise the loading path instead of the one under test.
  await waitFor(() => expect(expertRequests).toBeGreaterThan(0));
}

let expertRequests = 0;

beforeEach(() => {
  expertRequests = 0;
  // Two ChatGPT accounts and no usable platform route: the one shape where
  // the client cannot pick a route on its own and would normally ask.
  testState.transports = [unconfiguredSelfHosted, codexOne, codexTwo];
});

afterEach(() => {
  server.resetHandlers();
  testState.toast.mockClear();
  useCopilotUIStore.setState({ copilotLlmAuth: null });
});

function mockExpert(expert: Expert) {
  server.use(
    getGetExpertMockHandler(() => {
      expertRequests += 1;
      return expert;
    }),
  );
}

describe("useChatSession for an expert with a pinned connection", () => {
  it("names no route so the server starts the thread on the expert's pin", async () => {
    mockExpert(pinnedExpert);
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);
    await expertHasLoaded();

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({ expert_id: EXPERT_ID });
    });
    expect(testState.toast).not.toHaveBeenCalled();
  });

  it("defers to the server when the user sends before the pin has loaded", async () => {
    let releaseExpert: (expert: Expert) => void = () => {};
    server.use(
      getGetExpertMockHandler(() => {
        expertRequests += 1;
        return new Promise<Expert>((resolve) => {
          releaseExpert = resolve;
        });
      }),
    );
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);
    await waitFor(() => expect(expertRequests).toBeGreaterThan(0));

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({ expert_id: EXPERT_ID });
    });
    expect(testState.toast).not.toHaveBeenCalled();
    releaseExpert(pinnedExpert);
  });

  it("still sends a connection the user picked for this chat", async () => {
    mockExpert(pinnedExpert);
    useCopilotUIStore.getState().setCopilotLlmAuth({
      authProvider: "codex",
      credentialId: "codex-credential-1",
    });
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);
    await expertHasLoaded();

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() => {
      expect(getRequestBody()).toEqual({
        llm_auth_provider: "codex",
        llm_credential_id: "codex-credential-1",
        expert_id: EXPERT_ID,
      });
    });
  });

  it("asks for a choice when the expert follows an undecidable account default", async () => {
    mockExpert(unpinnedExpert);
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);
    await expertHasLoaded();

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() =>
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Choose an AI connection" }),
      ),
    );
    expect(getRequestBody()).toBeUndefined();
  });

  it("treats a pin to a missing connection as no pin", async () => {
    mockExpert({ ...pinnedExpert, llm_route_available: false });
    const getRequestBody = captureCreateRequest();
    render(<SessionHarness />);
    await expertHasLoaded();

    fireEvent.click(screen.getByRole("button", { name: "Create session" }));

    await waitFor(() =>
      expect(testState.toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Choose an AI connection" }),
      ),
    );
    expect(getRequestBody()).toBeUndefined();
  });
});
