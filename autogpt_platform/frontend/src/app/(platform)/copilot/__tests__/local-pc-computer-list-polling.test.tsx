import { getGetExperimentalListLocalExecutorsQueryKey } from "@/app/api/__generated__/endpoints/copilot/copilot";
import { getPostV2CreateSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetHomeDashboardMockHandler200,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { focusManager, type QueryClient } from "@tanstack/react-query";
import { act, cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import { budgetHandlers, renderChat } from "./budget-helpers";
import {
  sessionHandler,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
} from "./sse-helpers";

// The new chat page asks which computers are connected every 3 s. These count
// those requests on the real page, under a clock the test advances.

const PROXY = "http://localhost:3000/api/proxy/api";
const EXECUTORS_URL = `${PROXY}/copilot/executors`;
const DIRECTORIES_URL = `${EXECUTORS_URL}/machine-1/directories`;
const EXECUTOR_STATUS_URL = `${PROXY}/copilot/sessions/:sessionId/executor`;
const CREATE_CHAT_URL = `${PROXY}/chat/sessions`;
const STREAM_URL = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
const LIST_QUERY_KEY = getGetExperimentalListLocalExecutorsQueryKey();
const HALF_A_MINUTE = 30_000;
// One every 3 s, less one for where the half minute starts.
const POLLED_THROUGHOUT = 9;
const MACHINE = {
  machine_id: "machine-1",
  connection_id: "connection-1",
  display_name: "Workstation",
  platform: "windows",
  arch: "x86_64",
  shim_version: "0.2.0",
  capabilities: ["files", "shell"],
};
const PROJECTS = {
  directory_ref: "projects-ref",
  name: "Projects",
  path: "C:\\Users\\Ada\\Projects",
};
const ROOM_LEFT = {
  tier: "PRO",
  daily: null,
  weekly: null,
  reset_cost: 0,
  state: "ok",
  limiting_window: null,
  resets_at: null,
} as const;

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
  useAuth: () => ({
    user: { id: "user-1", email: "ada@example.com" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  function isLocalPC(flag: string) {
    return flag === actual.Flag.LOCAL_PC_EXECUTOR;
  }
  return {
    ...actual,
    useGetFlag: isLocalPC,
    useFlagStatus: (flag: string) => ({
      enabled: isLocalPC(flag),
      ready: true,
      answered: true,
    }),
  };
});

const listRequests = vi.fn();
let connected = [MACHINE];
let client: QueryClient;

function renderPage(searchParams = "") {
  client = renderChat(searchParams).queryClient;
}

function resetTarget() {
  useCopilotUIStore.setState({
    newChatExecutionTarget: { kind: "cloud" },
    isExecutionTargetPickerOpen: false,
    executionTargetError: null,
  });
}

async function advance(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

/** How many times the list is asked for over the next half minute. */
async function requestsOverHalfAMinute() {
  // A request already on its way when the page changed is not a poll.
  await waitFor(() =>
    expect(client.isFetching({ queryKey: LIST_QUERY_KEY })).toBe(0),
  );
  const before = listRequests.mock.calls.length;
  await advance(HALF_A_MINUTE);
  return listRequests.mock.calls.length - before;
}

function newUser() {
  return userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
}

type User = ReturnType<typeof newUser>;

function folderDialog() {
  return screen.findByRole("dialog", {
    name: /^Choose a Folder on Your Local PC$/i,
  });
}

async function chooseLocalPC(user: User) {
  await user.click(
    await screen.findByRole("button", { name: "Execution target: Cloud" }),
  );
  await user.click(
    await screen.findByRole("menuitemradio", { name: /^Local PC/i }),
  );
  return folderDialog();
}

async function chooseTheProjectsFolder(user: User) {
  await chooseLocalPC(user);
  await user.click(await screen.findByRole("button", { name: "Projects" }));
  await user.click(
    await screen.findByRole("button", { name: "Use This Folder" }),
  );
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
}

async function closeTheFolderDialog(user: User) {
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
}

async function send(user: User, message: string) {
  const input = screen.getByLabelText(/chat message input/i);
  await waitFor(() => expect(input).toHaveProperty("disabled", false));
  await user.type(input, message);
  await user.click(screen.getByRole("button", { name: /^submit$/i }));
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  focusManager.setFocused(true);
  listRequests.mockReset();
  connected = [MACHINE];
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  sessionStorage.clear();
  resetTarget();
  storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
  server.use(
    ...budgetHandlers({ usage: ROOM_LEFT }),
    getGetHomeDashboardMockHandler200(
      getGetHomeDashboardResponseMock200({
        timezone: "UTC",
        attention: [],
        active_tasks: [],
        upcoming_tasks: [],
        agents: [],
      }),
    ),
    http.get(EXECUTORS_URL, () => {
      listRequests();
      return HttpResponse.json({ executors: connected });
    }),
    http.post(DIRECTORIES_URL, async ({ request }) => {
      const body = (await request.json()) as {
        expected_connection_id: string;
        directory_ref: string | null;
      };
      if (body.expected_connection_id !== connected[0]?.connection_id) {
        return HttpResponse.json(
          { detail: "The Local PC executor reconnected" },
          { status: 409 },
        );
      }
      const opened = body.directory_ref === PROJECTS.directory_ref;
      return HttpResponse.json({
        connection_id: body.expected_connection_id,
        browse_id: "browse-1",
        current: opened ? PROJECTS : null,
        parent_ref: null,
        entries: opened ? [] : [PROJECTS],
        truncated: false,
        expires_at: "2026-07-10T18:00:00Z",
      });
    }),
  );
});

afterEach(() => {
  cleanup();
  client.clear();
  focusManager.setFocused(undefined);
  vi.useRealTimers();
  storage.clean(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
  resetTarget();
});

describe("the computer list on the new chat page", () => {
  test("is asked for only while the folder dialog is open", async () => {
    const user = newUser();
    renderPage();
    await screen.findByRole("button", { name: "Execution target: Cloud" });

    expect(await requestsOverHalfAMinute()).toBe(0);

    await chooseLocalPC(user);
    expect(await requestsOverHalfAMinute()).toBeGreaterThanOrEqual(
      POLLED_THROUGHOUT,
    );

    // Local PC stays selected, with no folder: nobody is choosing or waiting.
    await closeTheFolderDialog(user);
    expect(await requestsOverHalfAMinute()).toBe(0);
  });

  test("stops once a folder is chosen, and starts again when the dialog reopens", async () => {
    const user = newUser();
    renderPage();

    await chooseTheProjectsFolder(user);
    expect(await requestsOverHalfAMinute()).toBe(0);

    await user.click(
      screen.getByRole("button", { name: /^Local folder: .*Change folder$/ }),
    );
    await folderDialog();
    expect(await requestsOverHalfAMinute()).toBeGreaterThanOrEqual(
      POLLED_THROUGHOUT,
    );
  });

  test("keeps being asked for while the dialog waits for a computer to connect", async () => {
    connected = [];
    const user = newUser();
    renderPage();

    await chooseLocalPC(user);
    await screen.findByText("Waiting for a signed-in executor…");

    expect(await requestsOverHalfAMinute()).toBeGreaterThanOrEqual(
      POLLED_THROUGHOUT,
    );
  });

  test("is not asked for after switching back to Cloud", async () => {
    const user = newUser();
    renderPage();

    await chooseLocalPC(user);
    await closeTheFolderDialog(user);
    await user.click(
      screen.getByRole("button", { name: "Execution target: Local PC" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: /^Cloud/i }),
    );
    await screen.findByRole("button", { name: "Execution target: Cloud" });

    expect(await requestsOverHalfAMinute()).toBe(0);
  });

  test("a computer that reconnected after its folder was chosen is caught when the chat is sent", async () => {
    // With the dialog closed nothing watches the computer, so the server's
    // refusal is what tells the person, and the dialog opens on the list again.
    const user = newUser();
    renderPage();
    await chooseTheProjectsFolder(user);
    connected = [{ ...MACHINE, connection_id: "connection-2" }];
    const sent = vi.fn();
    server.use(
      http.post(CREATE_CHAT_URL, async ({ request }) => {
        sent(await request.json());
        return HttpResponse.json(
          { detail: "The Local PC executor reconnected" },
          { status: 409 },
        );
      }),
    );

    await send(user, "Tidy my Projects folder");

    await folderDialog();
    expect(
      await screen.findByText(
        "Your Local PC reconnected. Choose the folder again.",
      ),
    ).toBeDefined();
    expect(sent).toHaveBeenCalledOnce();
    expect(sent.mock.calls[0][0]).toMatchObject({
      execution_target: {
        kind: "local",
        machine_id: "machine-1",
        expected_connection_id: "connection-1",
        directory_ref: "projects-ref",
      },
    });
    expect(await requestsOverHalfAMinute()).toBeGreaterThanOrEqual(
      POLLED_THROUGHOUT,
    );
    expect(useCopilotUIStore.getState().newChatExecutionTarget).toMatchObject({
      machineID: "machine-1",
      connectionID: "connection-2",
      directoryRef: null,
    });
    // The message is still theirs to send once the folder is chosen again.
    expect(screen.getByLabelText(/chat message input/i)).toHaveProperty(
      "value",
      "Tidy my Projects folder",
    );
  });

  test("is not asked for once the chat has started", async () => {
    const user = newUser();
    server.use(
      getPostV2CreateSessionMockHandler200({
        id: TEST_SESSION_ID,
        created_at: "2026-11-17T00:00:00Z",
        user_id: "user-1",
      }),
      sessionHandler({
        metadata: {
          dry_run: false,
          builder_graph_id: null,
          execution_target: {
            kind: "local",
            machine_id: "machine-1",
            allowed_root: PROJECTS.path,
            revision: 1,
          },
        },
      }),
      http.get(EXECUTOR_STATUS_URL, () =>
        HttpResponse.json({ kind: "none", computer_use_consent: "pending" }),
      ),
      http.post(STREAM_URL, ({ request }) =>
        streamSseResponse(assistantTextChunks("On it."), {
          abortSignal: request.signal,
        }),
      ),
    );
    renderPage();
    await chooseTheProjectsFolder(user);

    await send(user, "Tidy my Projects folder");
    await screen.findByText("On it.");
    await screen.findByRole("button", { name: /Open Local PC details$/ });

    expect(await requestsOverHalfAMinute()).toBe(0);
  });
});
