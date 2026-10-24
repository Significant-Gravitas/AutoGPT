import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  markKickoffDone,
  markKickoffPending,
} from "@/app/(platform)/copilot/expertKickoff";
import {
  getGetV2GetSessionMockHandler200,
  getGetV2GetSessionResponseMock200,
  getGetV2ListSessionsMockHandler200,
  getPostV2CancelSessionTaskMockHandler,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListWorkspaceFilesMockHandler200,
  getListWorkspaceFoldersMockHandler200,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, ws } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertChatDrawer } from "../ExpertChatDrawer";

const USER_ID = "user-1";
const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
const SESSION_ID = "session-zara";
const FRESH_SESSION_ID = "session-zara-fresh";
const KICKOFF_SESSION_ID = "session-zara-kickoff";
const RESET_SESSION_ID = "session-zara-reset";
const WORKSPACE_REPLY_SESSION_ID = "session-zara-workspace-reply";
const OLD_TASK_SESSION_ID = "session-zara-old-task";
const NEW_TASK_SESSION_ID = "session-zara-new-task";
const DISCONNECT_SESSION_ID = "session-zara-disconnect";
const RECONNECT_SEND_SESSION_ID = "session-zara-reconnect-send";
const CLOSED_SESSION_ID = "session-zara-closed";
const SAME_CAPTION_SESSION_ID = "session-zara-same-caption";
const STOP_SESSION_ID = "session-zara-stop";
const UPLOADED_FILE_ID = "7a1c0f7e-9f30-4a3b-a6a1-0000000000aa";
const WORKSPACE_FILE_ID = "7a1c0f7e-9f30-4a3b-a6a1-0000000000bb";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

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
    useGetFlag: (flag: Parameters<typeof actual.useGetFlag>[0]) => {
      const value = actual.useGetFlag(flag);
      return flag === actual.Flag.CHAT_WORKSPACE_FILES ? true : value;
    },
  };
});

const backendSocket = ws.link("ws://localhost:8001/ws");

const ZARA = {
  expertId: EXPERT_ID,
  name: "Zara",
  role: "GTM Strategist",
  avatarUrl: null,
};

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

interface Recorder {
  streamBodies: Array<Record<string, unknown>>;
  uploadSessionIds: Array<string | null>;
  pendingPosts: number;
}

function newRecorder(): Recorder {
  return { streamBodies: [], uploadSessionIds: [], pendingPosts: 0 };
}

function uploadHandler(recorder: Recorder) {
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/workspace/files/upload`,
    ({ request }) => {
      recorder.uploadSessionIds.push(
        new URL(request.url).searchParams.get("session_id"),
      );
      return HttpResponse.json({
        file_id: UPLOADED_FILE_ID,
        name: "photo.png",
        path: "/photo.png",
        mime_type: "image/png",
        size_bytes: 2048,
      });
    },
  );
}

function pendingHandler(recorder: Recorder, sessionId: string) {
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${sessionId}/messages/pending`,
    () => {
      recorder.pendingPosts += 1;
      return HttpResponse.json({ buffer_length: 1 });
    },
  );
}

function streamHandler(
  recorder: Recorder,
  sessionId: string,
  holds: Array<ReturnType<typeof createStreamHold>["hold"] | undefined> = [],
) {
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${sessionId}/stream`,
    async ({ request }) => {
      const turn = recorder.streamBodies.length;
      recorder.streamBodies.push(
        (await request.clone().json()) as Record<string, unknown>,
      );
      return streamSseResponse(assistantTextChunks(`Reply ${turn + 1}`), {
        abortSignal: request.signal,
        hold: holds[turn],
      });
    },
  );
}

function sessionHandler(sessionId: string) {
  return getGetV2GetSessionMockHandler200(
    getGetV2GetSessionResponseMock200({
      id: sessionId,
      expert_id: EXPERT_ID,
      messages: [],
      active_stream: null,
    }),
  );
}

function existingThreadList(sessionId = SESSION_ID) {
  return getGetV2ListSessionsMockHandler200({
    sessions: [
      {
        id: sessionId,
        created_at: "2026-09-27T18:00:00Z",
        updated_at: "2026-09-27T18:01:00Z",
        is_processing: false,
        expert_id: EXPERT_ID,
      },
    ],
    total: 1,
  });
}

function photo() {
  return new File(["x".repeat(2048)], "photo.png", { type: "image/png" });
}

// Role queries here run inside the composer, the open menu or the picker,
// never across the whole drawer: after each DOM change a page-wide role query
// makes happy-dom recompute every element's style, which under load can
// outlast findBy's default 1 s and starve the reply the test is waiting for.
function composer() {
  const form = screen.getByPlaceholderText("Message Zara…").closest("form");
  if (!form) throw new Error("The composer has no form");
  return form;
}

function findStopButton() {
  return waitFor(() =>
    within(composer()).getByRole("button", { name: /Stop/ }),
  );
}

// Once a reply ends, the chat waits 500 ms (FINISH_REFETCH_SETTLE_MS in
// useCopilotStream) and re-reads the session before a held message may go,
// so the next turn never starts within waitFor's default 1 s on a busy runner.
function waitForSecondTurn(recorder: Recorder) {
  return waitFor(() => expect(recorder.streamBodies.length).toBe(2), {
    timeout: 3_000,
  });
}

function clickQueueMessage() {
  fireEvent.click(
    within(composer()).getByRole("button", { name: "Queue message" }),
  );
}

function openAttachMenu() {
  const plus = within(composer()).getByRole("button", {
    name: "Add files and more",
  }) as HTMLButtonElement;
  expect(plus.disabled).toBe(false);
  fireEvent.pointerDown(plus, { button: 0 });
  const menu = document.getElementById(
    plus.getAttribute("aria-controls") ?? "",
  );
  if (!menu) throw new Error("The attach menu did not open");
  return menu;
}

async function attachLocalFile(user: ReturnType<typeof userEvent.setup>) {
  const menu = openAttachMenu();
  fireEvent.click(
    await within(menu).findByRole("menuitem", { name: /Attach file/ }),
  );
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  if (!input) throw new Error("The composer has no file input");
  await user.upload(input, photo());
  await screen.findByText("photo.png");
}

async function pickWorkspaceFile() {
  const menu = openAttachMenu();
  fireEvent.click(
    await within(menu).findByRole("menuitem", {
      name: /Use File from Workspace/,
    }),
  );
  const picker = screen.getByRole("dialog", {
    name: "Use a file from your workspace",
  });
  fireEvent.click(
    await within(picker).findByRole("button", { name: /brief\.pdf/ }),
  );
  fireEvent.click(within(picker).getByRole("button", { name: "Add 1 file" }));
  await screen.findByText("brief.pdf");
}

function workspaceFileHandlers() {
  return [
    getListWorkspaceFoldersMockHandler200({ folders: [] }),
    getListWorkspaceFilesMockHandler200({
      files: [
        {
          id: WORKSPACE_FILE_ID,
          name: "brief.pdf",
          path: "/brief.pdf",
          mime_type: "application/pdf",
          size_bytes: 4096,
          created_at: "2026-09-27T18:00:00Z",
          origin: "uploaded",
        },
      ],
      has_more: false,
    }),
  ];
}

async function enabledComposer() {
  const input = (await screen.findByPlaceholderText(
    "Message Zara…",
  )) as HTMLTextAreaElement;
  await waitFor(() => expect(input.disabled).toBe(false));
  return input;
}

function pendingBubble() {
  return screen.getByTestId("pending-upload-message");
}

beforeEach(() => {
  window.localStorage.clear();
  useCopilotStreamStore.getState().resetAll();
  server.use(backendSocket.addEventListener("connection", () => {}));
  useAuthStore.setState({
    user: { id: USER_ID, email: "zara-owner@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
});

afterEach(() => {
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

describe("ExpertChatDrawer attachments", () => {
  test("shows the message at once and sends a local file in a new thread", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const user = userEvent.setup();
    const recorder = newRecorder();
    const createRequest = deferred();
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
      http.post("/api/proxy/api/chat/sessions", async () => {
        await createRequest.promise;
        return HttpResponse.json(
          getPostV2CreateSessionResponseMock200({ id: FRESH_SESSION_ID }),
        );
      }),
      sessionHandler(FRESH_SESSION_ID),
      uploadHandler(recorder),
      streamHandler(recorder, FRESH_SESSION_ID),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    const input = await enabledComposer();
    await attachLocalFile(user);
    await user.type(input, "what is this{Enter}");

    const bubble = await screen.findByTestId("pending-upload-message");
    expect(within(bubble).getByText("what is this")).toBeDefined();
    expect(
      within(bubble).getByRole("img", { name: "photo.png" }),
    ).toBeDefined();
    expect(screen.queryByText("What can I do for you?")).toBeNull();
    expect(recorder.uploadSessionIds).toEqual([]);

    createRequest.resolve();

    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    expect(recorder.uploadSessionIds).toEqual([FRESH_SESSION_ID]);
    expect(recorder.streamBodies[0]).toMatchObject({
      message: "what is this",
      file_ids: [UPLOADED_FILE_ID],
    });
    expect(await screen.findByText("Reply 1")).toBeDefined();
  });

  test("sends a workspace file picked in an existing thread", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    server.use(
      existingThreadList(),
      sessionHandler(SESSION_ID),
      ...workspaceFileHandlers(),
      streamHandler(recorder, SESSION_ID),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await pickWorkspaceFile();
    await user.type(input, "summarize this{Enter}");

    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    expect(recorder.streamBodies[0]).toMatchObject({
      message: "summarize this",
      file_ids: [WORKSPACE_FILE_ID],
    });
  });

  test("holds a file sent mid-reply and sends it as the next turn", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const firstReply = createStreamHold(3);
    server.use(
      existingThreadList(),
      sessionHandler(SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, SESSION_ID),
      streamHandler(recorder, SESSION_ID, [firstReply.hold]),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await user.type(input, "hello{Enter}");
    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    await findStopButton();

    await attachLocalFile(user);
    await user.type(input, "and this");
    clickQueueMessage();

    const bubble = await screen.findByTestId("pending-upload-message");
    expect(within(bubble).getByText("and this")).toBeDefined();
    expect(
      within(bubble).getByRole("img", { name: "photo.png" }),
    ).toBeDefined();
    await waitFor(() =>
      expect(recorder.uploadSessionIds).toEqual([SESSION_ID]),
    );
    expect(recorder.streamBodies.length).toBe(1);

    firstReply.release();

    await waitForSecondTurn(recorder);
    expect(recorder.pendingPosts).toBe(0);
    expect(recorder.streamBodies[1]).toMatchObject({
      message: "and this",
      file_ids: [UPLOADED_FILE_ID],
    });
  });

  test("sends a file held mid-reply even when its caption repeats the last message", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const firstReply = createStreamHold(3);
    server.use(
      existingThreadList(SAME_CAPTION_SESSION_ID),
      sessionHandler(SAME_CAPTION_SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, SAME_CAPTION_SESSION_ID),
      streamHandler(recorder, SAME_CAPTION_SESSION_ID, [firstReply.hold]),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await user.type(input, "summarize this{Enter}");
    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    await findStopButton();

    await attachLocalFile(user);
    await user.type(input, "summarize this");
    clickQueueMessage();
    await waitFor(() =>
      expect(recorder.uploadSessionIds).toEqual([SAME_CAPTION_SESSION_ID]),
    );

    firstReply.release();

    await waitForSecondTurn(recorder);
    expect(recorder.streamBodies[1]).toMatchObject({
      message: "summarize this",
      file_ids: [UPLOADED_FILE_ID],
    });
  });

  test("gives a follow-up waiting on the reply back when the user stops the reply", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const firstReply = createStreamHold(3);
    let cancels = 0;
    server.use(
      existingThreadList(STOP_SESSION_ID),
      sessionHandler(STOP_SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, STOP_SESSION_ID),
      streamHandler(recorder, STOP_SESSION_ID, [firstReply.hold]),
      getPostV2CancelSessionTaskMockHandler(() => {
        cancels += 1;
        return { cancelled: true };
      }),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await user.type(input, "hello{Enter}");
    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    await attachLocalFile(user);
    await user.type(input, "and this");
    clickQueueMessage();
    await waitFor(() =>
      expect(recorder.uploadSessionIds).toEqual([STOP_SESSION_ID]),
    );

    const stop = await findStopButton();
    await waitFor(
      () => expect((stop as HTMLButtonElement).disabled).toBe(false),
      {
        timeout: 2000,
      },
    );
    fireEvent.click(stop);

    await waitFor(() => expect(cancels).toBe(1));
    await waitFor(() =>
      expect(
        (screen.getByPlaceholderText("Message Zara…") as HTMLTextAreaElement)
          .value,
      ).toBe("and this"),
    );
    expect(screen.queryByTestId("pending-upload-message")).toBeNull();
    expect(recorder.streamBodies.length).toBe(1);
    expect(recorder.pendingPosts).toBe(0);
  });

  test("gives a follow-up waiting on the reply back when the user starts a new task", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const firstReply = createStreamHold(3);
    server.use(
      existingThreadList(RESET_SESSION_ID),
      sessionHandler(RESET_SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, RESET_SESSION_ID),
      streamHandler(recorder, RESET_SESSION_ID, [firstReply.hold]),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await user.type(input, "hello{Enter}");
    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    await attachLocalFile(user);
    await user.type(input, "and this{Enter}");
    await screen.findByTestId("pending-upload-message");

    fireEvent.click(screen.getByRole("button", { name: "New task" }));
    firstReply.release();

    await waitFor(() =>
      expect(
        (screen.getByPlaceholderText("Message Zara…") as HTMLTextAreaElement)
          .value,
      ).toBe("and this"),
    );
    expect(screen.queryByTestId("pending-upload-message")).toBeNull();
    expect(recorder.streamBodies.length).toBe(1);
    expect(recorder.pendingPosts).toBe(0);
  });

  test("sends a file queued behind the kickoff once the kickoff replies", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const kickoffReply = createStreamHold(3);
    let createAttempts = 0;
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
      http.post("/api/proxy/api/chat/sessions", () => {
        createAttempts += 1;
        if (createAttempts === 1) {
          return HttpResponse.json(
            { detail: "Could not create session" },
            { status: 500 },
          );
        }
        return HttpResponse.json(
          getPostV2CreateSessionResponseMock200({ id: KICKOFF_SESSION_ID }),
        );
      }),
      sessionHandler(KICKOFF_SESSION_ID),
      uploadHandler(recorder),
      streamHandler(recorder, KICKOFF_SESSION_ID, [kickoffReply.hold]),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    await waitFor(() => expect(createAttempts).toBe(1));
    const input = await enabledComposer();
    await attachLocalFile(user);
    await user.type(input, "what is this{Enter}");

    const bubble = await screen.findByTestId("pending-upload-message");
    expect(within(bubble).getByText("what is this")).toBeDefined();

    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    expect(recorder.streamBodies[0].expert_kickoff).toBe(true);
    expect(
      within(pendingBubble()).getByRole("img", { name: "photo.png" }),
    ).toBeDefined();

    kickoffReply.release();

    await waitForSecondTurn(recorder);
    expect(recorder.uploadSessionIds).toEqual([KICKOFF_SESSION_ID]);
    expect(recorder.streamBodies[1]).toMatchObject({
      expert_kickoff: false,
      message: "what is this",
      file_ids: [UPLOADED_FILE_ID],
    });
  });
  test("sends a workspace file picked mid-reply as the next turn", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const firstReply = createStreamHold(3);
    server.use(
      existingThreadList(WORKSPACE_REPLY_SESSION_ID),
      sessionHandler(WORKSPACE_REPLY_SESSION_ID),
      ...workspaceFileHandlers(),
      pendingHandler(recorder, WORKSPACE_REPLY_SESSION_ID),
      streamHandler(recorder, WORKSPACE_REPLY_SESSION_ID, [firstReply.hold]),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    await user.type(input, "hello{Enter}");
    await findStopButton();

    await pickWorkspaceFile();
    await user.type(input, "and this one{Enter}");

    const bubble = await screen.findByTestId("pending-upload-message");
    expect(within(bubble).getByText("brief.pdf")).toBeDefined();
    expect(recorder.streamBodies.length).toBe(1);

    firstReply.release();

    await waitForSecondTurn(recorder);
    expect(recorder.pendingPosts).toBe(0);
    expect(recorder.streamBodies[1]).toMatchObject({
      message: "and this one",
      file_ids: [WORKSPACE_FILE_ID],
    });
  });

  test("never sends a message into a new task when its upload outlives its thread", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const user = userEvent.setup();
    const recorder = newRecorder();
    const upload = deferred();
    const sentTo: string[] = [];
    server.use(
      existingThreadList(OLD_TASK_SESSION_ID),
      sessionHandler(OLD_TASK_SESSION_ID),
      sessionHandler(NEW_TASK_SESSION_ID),
      http.post("/api/proxy/api/chat/sessions", () =>
        HttpResponse.json(
          getPostV2CreateSessionResponseMock200({ id: NEW_TASK_SESSION_ID }),
        ),
      ),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/workspace/files/upload`,
        async () => {
          await upload.promise;
          return HttpResponse.json({
            file_id: UPLOADED_FILE_ID,
            name: "photo.png",
            mime_type: "image/png",
          });
        },
      ),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/:sessionId/stream`,
        async ({ params, request }) => {
          sentTo.push(String(params.sessionId));
          recorder.streamBodies.push(
            (await request.clone().json()) as Record<string, unknown>,
          );
          return streamSseResponse(assistantTextChunks("New task reply"), {
            abortSignal: request.signal,
          });
        },
      ),
    );

    try {
      render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

      const input = await enabledComposer();
      await attachLocalFile(user);
      await user.type(input, "old task attachment{Enter}");
      await screen.findByTestId("pending-upload-message");

      fireEvent.click(screen.getByRole("button", { name: "New task" }));
      const newTaskInput = await enabledComposer();
      await user.type(newTaskInput, "new task question{Enter}");
      await screen.findByText("New task reply");
      await user.type(newTaskInput, "new unsent draft");

      upload.resolve();

      await waitFor(() =>
        expect(newTaskInput.value).toBe(
          "old task attachment\n\nnew unsent draft",
        ),
      );
      expect(sentTo).toEqual([NEW_TASK_SESSION_ID]);
      expect(recorder.streamBodies).toEqual([
        expect.objectContaining({ message: "new task question" }),
      ]);
    } finally {
      upload.resolve();
    }
  });

  test("keeps a held file through a dropped connection and sends it once the reply has resumed and ended", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const disconnect = deferred();
    const resumed = createStreamHold(5);
    const encoder = new TextEncoder();
    server.use(
      existingThreadList(DISCONNECT_SESSION_ID),
      sessionHandler(DISCONNECT_SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, DISCONNECT_SESSION_ID),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${DISCONNECT_SESSION_ID}/stream`,
        async ({ request }) => {
          const turn = recorder.streamBodies.length;
          recorder.streamBodies.push(
            (await request.clone().json()) as Record<string, unknown>,
          );
          if (turn > 0) {
            return streamSseResponse(assistantTextChunks("Second reply"), {
              abortSignal: request.signal,
            });
          }
          return new HttpResponse(
            new ReadableStream<Uint8Array>({
              async start(controller) {
                for (const chunk of assistantTextChunks("First reply").slice(
                  0,
                  4,
                )) {
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`),
                  );
                }
                await disconnect.promise;
                controller.error(new TypeError("network failed"));
              },
            }),
            {
              headers: {
                "content-type": "text/event-stream",
                "x-vercel-ai-ui-message-stream": "v1",
              },
            },
          );
        },
      ),
      http.get(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${DISCONNECT_SESSION_ID}/stream`,
        ({ request }) =>
          streamSseResponse(assistantTextChunks("Resumed reply"), {
            hold: resumed.hold,
            abortSignal: request.signal,
          }),
      ),
    );

    try {
      render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

      const input = await enabledComposer();
      await user.type(input, "hello{Enter}");
      await findStopButton();
      await attachLocalFile(user);
      await user.type(input, "attachment after reply{Enter}");
      await screen.findByTestId("pending-upload-message");
      await waitFor(() =>
        expect(recorder.uploadSessionIds).toEqual([DISCONNECT_SESSION_ID]),
      );

      disconnect.resolve();

      await screen.findByText("Resumed reply", undefined, { timeout: 5_000 });
      expect(screen.getByTestId("pending-upload-message")).toBeDefined();
      expect(recorder.streamBodies.length).toBe(1);

      resumed.release();

      await waitFor(() => expect(recorder.streamBodies.length).toBe(2), {
        timeout: 5_000,
      });
      expect(recorder.pendingPosts).toBe(0);
      expect(recorder.streamBodies[1]).toMatchObject({
        message: "attachment after reply",
        file_ids: [UPLOADED_FILE_ID],
      });
    } finally {
      disconnect.resolve();
      resumed.release();
    }
  }, 20_000);

  test("holds a file sent while the connection is being restored and sends it once the resumed reply ends", async () => {
    const user = userEvent.setup();
    const recorder = newRecorder();
    const disconnect = deferred();
    const resumed = createStreamHold(5);
    const encoder = new TextEncoder();
    server.use(
      existingThreadList(RECONNECT_SEND_SESSION_ID),
      sessionHandler(RECONNECT_SEND_SESSION_ID),
      uploadHandler(recorder),
      pendingHandler(recorder, RECONNECT_SEND_SESSION_ID),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${RECONNECT_SEND_SESSION_ID}/stream`,
        async ({ request }) => {
          const turn = recorder.streamBodies.length;
          recorder.streamBodies.push(
            (await request.clone().json()) as Record<string, unknown>,
          );
          if (turn > 0) {
            return streamSseResponse(assistantTextChunks("Second reply"), {
              abortSignal: request.signal,
            });
          }
          return new HttpResponse(
            new ReadableStream<Uint8Array>({
              async start(controller) {
                for (const chunk of assistantTextChunks("First reply").slice(
                  0,
                  4,
                )) {
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`),
                  );
                }
                await disconnect.promise;
                controller.error(new TypeError("network failed"));
              },
            }),
            {
              headers: {
                "content-type": "text/event-stream",
                "x-vercel-ai-ui-message-stream": "v1",
              },
            },
          );
        },
      ),
      http.get(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${RECONNECT_SEND_SESSION_ID}/stream`,
        ({ request }) =>
          streamSseResponse(assistantTextChunks("Resumed reply"), {
            hold: resumed.hold,
            abortSignal: request.signal,
          }),
      ),
    );

    try {
      render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

      const input = await enabledComposer();
      await user.type(input, "hello{Enter}");
      await findStopButton();
      await attachLocalFile(user);
      await user.type(input, "sent while reconnecting");

      disconnect.resolve();
      await waitFor(() =>
        expect(
          within(composer()).queryByRole("button", { name: /Stop/ }),
        ).toBeNull(),
      );
      await user.type(input, "{Enter}");

      await waitFor(() =>
        expect(recorder.uploadSessionIds).toEqual([RECONNECT_SEND_SESSION_ID]),
      );
      await screen.findByText("Resumed reply", undefined, { timeout: 5_000 });
      expect(pendingBubble().textContent).toContain("sent while reconnecting");
      expect(recorder.streamBodies).toEqual([
        expect.objectContaining({ message: "hello" }),
      ]);

      resumed.release();

      await waitFor(() => expect(recorder.streamBodies.length).toBe(2), {
        timeout: 5_000,
      });
      expect(recorder.pendingPosts).toBe(0);
      expect(recorder.streamBodies[1]).toMatchObject({
        message: "sent while reconnecting",
        file_ids: [UPLOADED_FILE_ID],
      });
    } finally {
      disconnect.resolve();
      resumed.release();
    }
  }, 20_000);

  test.each([
    { when: "while the expert replies", replying: true },
    { when: "with no reply in flight", replying: false },
  ])(
    "leaves the chat usable when the panel closes during an upload $when",
    async ({ replying }) => {
      markKickoffDone(
        USER_ID,
        EXPERT_ID,
        markKickoffPending(USER_ID, EXPERT_ID),
      );
      const user = userEvent.setup();
      const recorder = newRecorder();
      const upload = deferred();
      const reply = createStreamHold(3);
      let uploadReturned = false;
      server.use(
        existingThreadList(CLOSED_SESSION_ID),
        sessionHandler(CLOSED_SESSION_ID),
        streamHandler(recorder, CLOSED_SESSION_ID, [reply.hold]),
        http.post(
          `${TEST_BACKEND_BASE_URL}/api/workspace/files/upload`,
          async () => {
            recorder.uploadSessionIds.push(CLOSED_SESSION_ID);
            await upload.promise;
            uploadReturned = true;
            return HttpResponse.json({
              file_id: UPLOADED_FILE_ID,
              name: "photo.png",
              mime_type: "image/png",
            });
          },
        ),
      );

      try {
        const view = render(
          <ExpertChatDrawer target={ZARA} onClose={() => {}} />,
        );
        const input = await enabledComposer();
        if (replying) {
          await user.type(input, "hello{Enter}");
          await findStopButton();
        }
        await attachLocalFile(user);
        await user.type(input, "held while the upload runs{Enter}");
        await screen.findByTestId("pending-upload-message");
        await waitFor(() =>
          expect(recorder.uploadSessionIds).toEqual([CLOSED_SESSION_ID]),
        );

        view.unmount();
        upload.resolve();
        await waitFor(() => expect(uploadReturned).toBe(true));
        reply.release();

        render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

        await waitFor(() =>
          expect(
            useCopilotStreamStore.getState().pendingUploadSends[
              CLOSED_SESSION_ID
            ],
          ).toBeUndefined(),
        );
        await enabledComposer();
        expect(screen.queryByTestId("pending-upload-message")).toBeNull();
        expect(recorder.streamBodies).toEqual(
          replying ? [expect.objectContaining({ message: "hello" })] : [],
        );
      } finally {
        upload.resolve();
        reply.release();
      }
    },
  );
});
