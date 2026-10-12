import {
  getGetV2GetBoardMockHandler,
  getGetV2GetSessionMockHandler200,
  getPatchV2UpdateSessionBoardMockHandler,
  getPatchV2UpdateSessionExpertMockHandler,
  getPatchV2UpdateSessionExpertMockHandler409,
  getPatchV2UpdateSessionTitleMockHandler,
  getPostV2StartSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { BoardResponse } from "@/app/api/__generated__/models/boardResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { TaskBoardPage } from "../TaskBoardPage";

const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.HIRE_EXPERTS,
  };
});

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => "/home",
    useSearchParams: () => new URLSearchParams(),
  };
});

vi.mock("@/app/(platform)/copilot/CopilotChatHost", () => ({
  CopilotChatHost: () => <div>chat host</div>,
}));

const NOW = new Date().toISOString();
const S1 = "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b";
const ELSEWHERE = "8d7c6b5a-4f3e-4d2c-9b1a-0f9e8d7c6b5a";

const COLUMNS: BoardResponse["columns"] = [
  { key: "backlog", name: "Backlog", builtin: true },
  { key: "in_progress", name: "In progress", builtin: true },
  { key: "done", name: "Done", builtin: true },
];

function card(
  id: string,
  overrides: Partial<SessionSummaryResponse>,
): SessionSummaryResponse {
  return {
    id,
    created_at: NOW,
    updated_at: NOW,
    is_processing: false,
    started: true,
    file_count: 0,
    ...overrides,
  };
}

const DRAFT = card(S1, {
  title: "Write PRD",
  board_column: "backlog",
  board_position: 0,
  started: false,
});

const RUNNING = card(S1, {
  title: "Research competitors",
  board_column: "in_progress",
  board_position: 0,
});

const PRIYA = {
  id: "expert-priya",
  name: "Priya",
  avatar_url: null,
  role: "Product manager",
  is_archived: false,
};

function board(...sessions: SessionSummaryResponse[]): BoardResponse {
  return { columns: COLUMNS, sessions };
}

let boardRequests = 0;
let sessionRequests: string[] = [];

/** Answers the first board request with `first`, every later one with
 *  `later` — the refetch after a move. */
function serveBoard(first: BoardResponse, later: BoardResponse = first) {
  server.use(
    getGetV2GetBoardMockHandler(() => {
      boardRequests += 1;
      return boardRequests === 1 ? first : later;
    }),
  );
}

function captureBodies<T>() {
  const bodies: T[] = [];
  return {
    bodies,
    async push(request: Request) {
      bodies.push((await request.json()) as T);
    },
  };
}

function renderPanel(
  sessionId = S1,
  onUrlUpdate?: (event: UrlUpdateEvent) => void,
) {
  return render(
    <NuqsTestingAdapter
      searchParams={`?sessionId=${sessionId}`}
      onUrlUpdate={onUrlUpdate}
      hasMemory
    >
      <TaskBoardPage />
    </NuqsTestingAdapter>,
  );
}

function cardIn(column: string, title: string) {
  // The open sheet is modal, so the board behind it is aria-hidden.
  const region = screen.getByRole("region", { name: column, hidden: true });
  return within(region).queryByRole("article", { name: title, hidden: true });
}

function recordSessionRequests({ request }: { request: Request }) {
  const { pathname } = new URL(request.url);
  if (pathname.includes("/api/chat/sessions")) sessionRequests.push(pathname);
}

beforeEach(() => {
  boardRequests = 0;
  sessionRequests = [];
  toastSpy.mockReset();
  server.use(getListExpertIdentitiesMockHandler([PRIYA]));
  server.events.on("request:start", recordSessionRequests);
});

afterEach(() => {
  server.events.removeListener("request:start", recordSessionRequests);
  vi.useRealTimers();
});

describe("task panel", () => {
  test("opens a started card as a dialog with the chat", async () => {
    serveBoard(board(RUNNING));
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    expect(within(dialog).getByText("chat host")).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: /Reassign/ }),
    ).toBeDefined();
  });

  test("opens a session that is not on the board with its title and the chat", async () => {
    serveBoard(board(RUNNING));
    server.use(
      getGetV2GetSessionMockHandler200({
        id: ELSEWHERE,
        created_at: NOW,
        updated_at: NOW,
        user_id: "user-1",
        title: "Plan offsite",
        messages: [],
      }),
    );
    renderPanel(ELSEWHERE);

    const dialog = await screen.findByRole("dialog", { name: "Plan offsite" });
    expect(within(dialog).getByText("chat host")).toBeDefined();
    expect(
      within(dialog).queryByRole("combobox", { name: "Column" }),
    ).toBeNull();
    expect(
      within(dialog).queryByRole("button", { name: "Mark done" }),
    ).toBeNull();
  });

  test("ignores a sessionId that is not a UUID", async () => {
    serveBoard(board(RUNNING));
    renderPanel("../../x");

    expect(
      await screen.findByRole("region", { name: "In progress" }),
    ).toBeDefined();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(sessionRequests).toEqual([]);
  });

  test("starts a draft card with the first message and then shows the chat", async () => {
    const starts = captureBodies<Record<string, unknown>>();
    const started = { ...DRAFT, started: true, board_column: "in_progress" };
    serveBoard(board(DRAFT), board(started));
    server.use(
      getPostV2StartSessionMockHandler(async ({ request }) => {
        await starts.push(request);
        return started;
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    const message = within(dialog).getByRole("textbox", {
      name: "First message",
    });
    expect((message as HTMLTextAreaElement).value).toBe("Write PRD");
    expect(within(dialog).queryByText("chat host")).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Start" }),
    );

    await waitFor(() => expect(starts.bodies).toHaveLength(1));
    expect(starts.bodies[0]).toEqual({ message: "Write PRD" });
    expect(await within(dialog).findByText("chat host")).toBeDefined();
  });

  test("a not-started card with a greeting shows the chat and can start", async () => {
    const starts = captureBodies<Record<string, unknown>>();
    const greeted = { ...DRAFT, has_messages: true };
    const started = { ...greeted, started: true, board_column: "in_progress" };
    serveBoard(board(greeted), board(started));
    server.use(
      getPostV2StartSessionMockHandler(async ({ request }) => {
        await starts.push(request);
        return started;
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    expect(within(dialog).getByText("chat host")).toBeDefined();
    expect(
      within(dialog).queryByRole("textbox", { name: "First message" }),
    ).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Start" }),
    );

    await waitFor(() => expect(starts.bodies).toHaveLength(1));
    expect(starts.bodies[0]).toEqual({ message: "Write PRD" });
    await waitFor(() =>
      expect(
        within(dialog).queryByRole("button", { name: "Start" }),
      ).toBeNull(),
    );
    expect(within(dialog).getByText("chat host")).toBeDefined();
  });

  test("a start over the usage limit says why", async () => {
    serveBoard(board(DRAFT));
    server.use(
      http.post("*/api/chat/sessions/:sessionId/start", () =>
        HttpResponse.json(
          {
            detail: {
              kind: "usage_limit",
              message: "You've reached your daily limit",
            },
          },
          { status: 429 },
        ),
      ),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Start" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "You've reached your daily limit" }),
      ),
    );
    expect(
      within(dialog).getByRole("textbox", { name: "First message" }),
    ).toBeDefined();
  });

  test("a board poll during a move does not snap the card back", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const moves = captureBodies<Record<string, unknown>>();
    const moved = { ...RUNNING, board_column: "done", board_position: -1024 };
    let isMoveSaved = false;
    let releaseMove = () => {};
    const moveHeld = new Promise<void>((resolve) => {
      releaseMove = resolve;
    });
    server.use(
      getGetV2GetBoardMockHandler(() => {
        boardRequests += 1;
        return isMoveSaved ? board(moved) : board(RUNNING);
      }),
      getPatchV2UpdateSessionBoardMockHandler(async ({ request }) => {
        await moves.push(request);
        await moveHeld;
        isMoveSaved = true;
        return moved;
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    await user.click(within(dialog).getByRole("combobox", { name: "Column" }));
    await user.click(await screen.findByRole("option", { name: "Done" }));

    await waitFor(() => expect(moves.bodies).toHaveLength(1));
    expect(moves.bodies[0]).toEqual({ column: "done" });
    expect(cardIn("Done", "Research competitors")).not.toBeNull();

    // Past the 10 s poll while the PATCH is still open: a poll would serve
    // the stale board and put the card back in In progress.
    await vi.advanceTimersByTimeAsync(11_000);
    expect(boardRequests).toBe(1);
    expect(cardIn("Done", "Research competitors")).not.toBeNull();

    releaseMove();
    await waitFor(() => expect(boardRequests).toBeGreaterThan(1));
    expect(cardIn("Done", "Research competitors")).not.toBeNull();
    expect(cardIn("In progress", "Research competitors")).toBeNull();
  });

  test("moving a draft card to In progress says it started", async () => {
    const moves = captureBodies<Record<string, unknown>>();
    const moved = {
      ...DRAFT,
      board_column: "in_progress",
      board_position: -1024,
      started: true,
      is_processing: true,
    };
    serveBoard(board(DRAFT), board(moved));
    server.use(
      getPatchV2UpdateSessionBoardMockHandler(async ({ request }) => {
        await moves.push(request);
        return moved;
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("combobox", { name: "Column" }),
    );
    await userEvent.click(
      await screen.findByRole("option", { name: "In progress" }),
    );

    await waitFor(() => expect(moves.bodies).toHaveLength(1));
    expect(moves.bodies[0]).toEqual({ column: "in_progress" });
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Started" }),
      ),
    );
  });

  test("Mark done moves the card to Done and hides itself there", async () => {
    const moves = captureBodies<Record<string, unknown>>();
    const moved = { ...RUNNING, board_column: "done", board_position: -1024 };
    serveBoard(board(RUNNING), board(moved));
    server.use(
      getPatchV2UpdateSessionBoardMockHandler(async ({ request }) => {
        await moves.push(request);
        return moved;
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Mark done" }),
    );

    await waitFor(() => expect(moves.bodies).toHaveLength(1));
    expect(moves.bodies[0]).toEqual({ column: "done" });
    await waitFor(() =>
      expect(
        within(dialog).queryByRole("button", { name: "Mark done" }),
      ).toBeNull(),
    );
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Started" }),
    );
  });

  test("a failed move puts the card back and says so", async () => {
    serveBoard(board(RUNNING));
    server.use(
      http.patch("*/api/chat/sessions/:sessionId/board", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Mark done" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't move the task" }),
      ),
    );
    expect(cardIn("In progress", "Research competitors")).not.toBeNull();
  });

  test("a move refused by the paywall says why", async () => {
    serveBoard(board(DRAFT));
    server.use(
      http.patch("*/api/chat/sessions/:sessionId/board", () =>
        HttpResponse.json(
          { detail: "A subscription is required to use this feature." },
          { status: 402 },
        ),
      ),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("combobox", { name: "Column" }),
    );
    await userEvent.click(
      await screen.findByRole("option", { name: "In progress" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "A subscription is required to use this feature.",
        }),
      ),
    );
    expect(cardIn("Backlog", "Write PRD")).not.toBeNull();
  });

  test("assigns a draft card to an expert", async () => {
    const assigns = captureBodies<Record<string, unknown>>();
    serveBoard(board(DRAFT), board({ ...DRAFT, expert_id: "expert-priya" }));
    server.use(
      getPatchV2UpdateSessionExpertMockHandler(async ({ request }) => {
        await assigns.push(request);
        return { ...DRAFT, expert_id: "expert-priya" };
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("button", { name: /Assignee: Otto/ }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Priya/ }),
    );

    await waitFor(() => expect(assigns.bodies).toHaveLength(1));
    expect(assigns.bodies[0]).toEqual({ expert_id: "expert-priya" });
    expect(
      await within(dialog).findByRole("button", { name: /Assignee: Priya/ }),
    ).toBeDefined();
  });

  test("tells the user to reassign when the task already started", async () => {
    serveBoard(board(DRAFT));
    server.use(getPatchV2UpdateSessionExpertMockHandler409());
    renderPanel();

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("button", { name: /Assignee: Otto/ }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Priya/ }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "This task already started. Use Reassign instead.",
        }),
      ),
    );
    expect(
      within(dialog).getByRole("button", { name: /Assignee: Otto/ }),
    ).toBeDefined();
  });

  test("renames the task with Enter and cancels with Escape", async () => {
    const renames = captureBodies<Record<string, unknown>>();
    serveBoard(board(RUNNING));
    server.use(
      getPatchV2UpdateSessionTitleMockHandler(async ({ request }) => {
        await renames.push(request);
        return {};
      }),
    );
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Research competitors" }),
    );
    const input = within(dialog).getByRole("textbox", { name: "Task title" });
    await userEvent.type(input, " again{Escape}");
    expect(renames.bodies).toHaveLength(0);
    expect(screen.getByRole("dialog")).toBeDefined();

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Research competitors" }),
    );
    await userEvent.clear(
      within(dialog).getByRole("textbox", { name: "Task title" }),
    );
    await userEvent.type(
      within(dialog).getByRole("textbox", { name: "Task title" }),
      "Research rivals{Enter}",
    );

    await waitFor(() => expect(renames.bodies).toHaveLength(1));
    expect(renames.bodies[0]).toEqual({ title: "Research rivals" });
    expect(
      await screen.findByRole("dialog", { name: "Research rivals" }),
    ).toBeDefined();
  });

  test("Escape in a text field leaves the panel open", async () => {
    serveBoard(board(DRAFT));
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderPanel(S1, onUrlUpdate);

    const dialog = await screen.findByRole("dialog", { name: "Write PRD" });
    await userEvent.click(
      within(dialog).getByRole("textbox", { name: "First message" }),
    );
    await userEvent.keyboard("{Escape}");

    expect(screen.getByRole("dialog", { name: "Write PRD" })).toBeDefined();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  test("Escape on the panel itself closes it", async () => {
    serveBoard(board(RUNNING));
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderPanel(S1, onUrlUpdate);

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    within(dialog).getByRole("button", { name: "Files" }).focus();
    await userEvent.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.has("sessionId")).toBe(
      false,
    );
  });

  test("Close clears the session from the URL", async () => {
    serveBoard(board(RUNNING));
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderPanel(S1, onUrlUpdate);

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Close" }),
    );

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.has("sessionId")).toBe(
      false,
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  test("Expand links to the full-page chat", async () => {
    serveBoard(board(RUNNING));
    renderPanel();

    const dialog = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    expect(
      within(dialog).getByRole("link", { name: "Expand" }).getAttribute("href"),
    ).toBe(`/chat?sessionId=${S1}`);
  });
});
