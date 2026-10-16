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
const RETRY_SESSION_ID = "session-zara-retry";
const REMOVE_SESSION_ID = "session-zara-remove";
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

const NOTES_FILE_ID = "7a1c0f7e-9f30-4a3b-a6a1-0000000000cc";

/** Fails the first upload of `notes.pdf`; every other upload lands. */
function uploadFailingNotesOnce(uploadedNames: string[]) {
  let hasFailed = false;
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/workspace/files/upload`,
    async ({ request }) => {
      const file = (await request.formData()).get("file") as File;
      uploadedNames.push(file.name);
      if (file.name === "notes.pdf" && !hasFailed) {
        hasFailed = true;
        return HttpResponse.json({ detail: "Upload failed" }, { status: 500 });
      }
      return HttpResponse.json({
        file_id: file.name === "notes.pdf" ? NOTES_FILE_ID : UPLOADED_FILE_ID,
        name: file.name,
        path: `/${file.name}`,
        mime_type: file.type,
        size_bytes: file.size,
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

async function attachLocalFile(user: ReturnType<typeof userEvent.setup>) {
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  if (!input) throw new Error("The composer has no file input");
  await user.upload(input, photo());
  await screen.findByText("photo.png");
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
    expect(within(bubble).getByText("photo.png")).toBeDefined();
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
      streamHandler(recorder, SESSION_ID),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    fireEvent.pointerDown(screen.getByTestId("composer-plus-button"), {
      button: 0,
    });
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /Use File from Workspace/ }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /brief\.pdf/ }));
    fireEvent.click(screen.getByRole("button", { name: "Add 1 file" }));
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

    await attachLocalFile(user);
    await user.type(input, "and this{Enter}");

    const bubble = await screen.findByTestId("pending-upload-message");
    expect(within(bubble).getByText("and this")).toBeDefined();
    expect(within(bubble).getByText("photo.png")).toBeDefined();
    await waitFor(() =>
      expect(recorder.uploadSessionIds).toEqual([SESSION_ID]),
    );
    expect(recorder.streamBodies.length).toBe(1);

    firstReply.release();

    await waitFor(() => expect(recorder.streamBodies.length).toBe(2));
    expect(recorder.pendingPosts).toBe(0);
    expect(recorder.streamBodies[1]).toMatchObject({
      message: "and this",
      file_ids: [UPLOADED_FILE_ID],
    });
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
    expect(within(pendingBubble()).getByText("photo.png")).toBeDefined();

    kickoffReply.release();

    await waitFor(() => expect(recorder.streamBodies.length).toBe(2));
    expect(recorder.uploadSessionIds).toEqual([KICKOFF_SESSION_ID]);
    expect(recorder.streamBodies[1]).toMatchObject({
      expert_kickoff: false,
      message: "what is this",
      file_ids: [UPLOADED_FILE_ID],
    });
  });
});

describe("ExpertChatDrawer failed uploads", () => {
  async function sendPhotoAndNotes(
    sessionId: string,
    recorder: Recorder,
    uploadedNames: string[],
  ) {
    const user = userEvent.setup();
    server.use(
      existingThreadList(sessionId),
      sessionHandler(sessionId),
      uploadFailingNotesOnce(uploadedNames),
      streamHandler(recorder, sessionId),
    );
    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    const input = await enabledComposer();
    const fileInput =
      document.querySelector<HTMLInputElement>('input[type="file"]');
    if (!fileInput) throw new Error("The composer has no file input");
    await user.upload(fileInput, [
      photo(),
      new File(["y".repeat(4096)], "notes.pdf", { type: "application/pdf" }),
    ]);
    await screen.findByText("notes.pdf");
    await user.type(input, "compare these{Enter}");

    const failedCard = await waitFor(() => {
      const card = within(pendingBubble())
        .getAllByTestId("pending-upload-attachment")
        .find((c) => within(c).queryByText("notes.pdf"));
      expect(card?.dataset.status).toBe("failed");
      return card!;
    });
    return { user, failedCard };
  }

  test("keeps a failed file in the bubble and sends nothing until it is retried", async () => {
    const recorder = newRecorder();
    const uploadedNames: string[] = [];
    const { user, failedCard } = await sendPhotoAndNotes(
      RETRY_SESSION_ID,
      recorder,
      uploadedNames,
    );

    expect(within(failedCard).getByText("Upload failed")).toBeDefined();
    expect(screen.getByRole("status").textContent).toBe(
      "Couldn't upload notes.pdf. Retry or remove it to send your message.",
    );
    const photoCard = within(pendingBubble())
      .getAllByTestId("pending-upload-attachment")
      .find((c) => within(c).queryByText("photo.png"));
    expect(photoCard?.dataset.status).toBe("uploaded");
    expect(recorder.streamBodies).toEqual([]);

    await user.click(
      within(failedCard).getByRole("button", { name: "Retry notes.pdf" }),
    );

    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    expect(uploadedNames.filter((name) => name === "photo.png")).toEqual([
      "photo.png",
    ]);
    expect(uploadedNames.filter((name) => name === "notes.pdf")).toEqual([
      "notes.pdf",
      "notes.pdf",
    ]);
    expect(recorder.streamBodies[0]).toMatchObject({
      message: "compare these",
      file_ids: [UPLOADED_FILE_ID, NOTES_FILE_ID],
    });
  });

  test("sends the rest once the failed file is removed", async () => {
    const recorder = newRecorder();
    const uploadedNames: string[] = [];
    const { user, failedCard } = await sendPhotoAndNotes(
      REMOVE_SESSION_ID,
      recorder,
      uploadedNames,
    );

    await user.click(
      within(failedCard).getByRole("button", { name: "Remove notes.pdf" }),
    );

    await waitFor(() => expect(recorder.streamBodies.length).toBe(1));
    expect(recorder.streamBodies[0]).toMatchObject({
      message: "compare these",
      file_ids: [UPLOADED_FILE_ID],
    });
    expect(uploadedNames.filter((name) => name === "notes.pdf")).toEqual([
      "notes.pdf",
    ]);
  });
});
