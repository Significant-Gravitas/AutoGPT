import {
  getGetV2GetBoardMockHandler,
  getPostV2CreateSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListWorkspaceFilesMockHandler } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
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
import { beforeEach, describe, expect, test, vi } from "vitest";
import { TaskBoardPage } from "../TaskBoardPage";

const flags = vi.hoisted(() => ({ hireExperts: true }));
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
    useGetFlag: (flag: string) =>
      flag === actual.Flag.HIRE_EXPERTS && flags.hireExperts,
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

const NOW = new Date().toISOString();

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

const BOARD: BoardResponse = {
  columns: COLUMNS,
  sessions: [
    card("s-draft", {
      title: "Draft launch plan",
      board_column: "backlog",
      board_position: 0,
      started: false,
    }),
    card("s-research", {
      title: "Research competitors",
      board_column: "in_progress",
      board_position: 0,
      is_processing: true,
      expert_id: "expert-priya",
    }),
    card("s-venue", {
      title: "Book venue",
      board_column: "in_progress",
      board_position: 1024,
      needs_you: true,
    }),
    card("s-ship", {
      title: "Ship v1",
      board_column: "done",
      board_position: 0,
      last_line: "Shipped the release notes.",
      file_count: 2,
    }),
  ],
};

const PRIYA = {
  id: "expert-priya",
  name: "Priya",
  avatar_url: null,
  role: "Product manager",
  is_archived: false,
};

let boardRequests: URL[] = [];
let createBodies: Record<string, unknown>[] = [];

function serveBoard(board: BoardResponse) {
  server.use(
    getGetV2GetBoardMockHandler(({ request }) => {
      boardRequests.push(new URL(request.url));
      return board;
    }),
  );
}

function renderBoard(onUrlUpdate?: (event: UrlUpdateEvent) => void) {
  return render(
    <NuqsTestingAdapter onUrlUpdate={onUrlUpdate}>
      <TaskBoardPage />
    </NuqsTestingAdapter>,
  );
}

async function findColumn(name: string) {
  return screen.findByRole("region", { name });
}

// Long enough for a stray POST (MSW answers after a 0 ms delay) to land.
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 50));
}

beforeEach(() => {
  flags.hireExperts = true;
  boardRequests = [];
  createBodies = [];
  toastSpy.mockReset();
  server.use(
    getListExpertIdentitiesMockHandler([PRIYA]),
    getPostV2CreateSessionMockHandler(async ({ request }) => {
      createBodies.push((await request.json()) as Record<string, unknown>);
      return { id: "s-new", created_at: NOW, user_id: "user-1" };
    }),
  );
});

describe("task board", () => {
  test("renders the columns with their cards, states and assignees", async () => {
    serveBoard(BOARD);
    renderBoard();

    const backlog = await findColumn("Backlog");
    const inProgress = await findColumn("In progress");
    const done = await findColumn("Done");

    expect(within(backlog).getByText("1")).toBeDefined();
    expect(within(inProgress).getByText("2")).toBeDefined();
    expect(within(done).getByText("1")).toBeDefined();

    const draft = within(backlog).getByRole("article", {
      name: "Draft launch plan",
    });
    expect(within(draft).getByText("Not started")).toBeDefined();
    expect(within(draft).getByText("Otto")).toBeDefined();

    const research = within(inProgress).getByRole("article", {
      name: "Research competitors",
    });
    expect(within(research).getByText("Working…")).toBeDefined();
    expect(await within(research).findByText("Priya")).toBeDefined();

    const venue = within(inProgress).getByRole("article", {
      name: "Book venue",
    });
    expect(within(venue).getByText("Needs you")).toBeDefined();
    expect(within(venue).getByText("Waiting for your answer")).toBeDefined();

    const ship = within(done).getByRole("article", { name: "Ship v1" });
    expect(within(ship).getByText("Shipped the release notes.")).toBeDefined();
  });

  test("names an expert's card when experts are off instead of loading forever", async () => {
    flags.hireExperts = false;
    serveBoard(BOARD);
    renderBoard();

    const inProgress = await findColumn("In progress");
    const research = within(inProgress).getByRole("article", {
      name: "Research competitors",
    });
    expect(within(research).getByText("This expert")).toBeDefined();
    expect(research.querySelector("span.bg-zinc-100[aria-hidden]")).toBeNull();
  });

  test("adds a task to Backlog and refreshes the board", async () => {
    serveBoard(BOARD);
    renderBoard();

    const backlog = await findColumn("Backlog");
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Add a task…" }),
    );
    const input = within(backlog).getByRole("textbox", { name: "New task" });
    const requestsBefore = boardRequests.length;
    await userEvent.type(input, "Write PRD{Enter}");

    await waitFor(() => expect(createBodies).toHaveLength(1));
    expect(createBodies[0]).toMatchObject({
      title: "Write PRD",
      board_column: "backlog",
      expert_id: null,
    });
    await waitFor(() =>
      expect(boardRequests.length).toBeGreaterThan(requestsBefore),
    );
    expect((input as HTMLInputElement).value).toBe("");
  });

  test("adds a task to In progress with that column", async () => {
    serveBoard(BOARD);
    renderBoard();

    const inProgress = await findColumn("In progress");
    await userEvent.click(
      within(inProgress).getByRole("button", { name: "Add a task…" }),
    );
    await userEvent.type(
      within(inProgress).getByRole("textbox", { name: "New task" }),
      "Call the printer{Enter}",
    );

    await waitFor(() => expect(createBodies).toHaveLength(1));
    expect(createBodies[0]).toMatchObject({
      title: "Call the printer",
      board_column: "in_progress",
    });
  });

  test("a task refused over the usage limit says why", async () => {
    serveBoard(BOARD);
    server.use(
      http.post("*/api/chat/sessions", () =>
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
    renderBoard();

    const inProgress = await findColumn("In progress");
    await userEvent.click(
      within(inProgress).getByRole("button", { name: "Add a task…" }),
    );
    await userEvent.type(
      within(inProgress).getByRole("textbox", { name: "New task" }),
      "Call the printer{Enter}",
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "You've reached your daily limit" }),
      ),
    );
  });

  test("does not add a blank task", async () => {
    serveBoard(BOARD);
    renderBoard();

    const backlog = await findColumn("Backlog");
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Add a task…" }),
    );
    const input = within(backlog).getByRole("textbox", { name: "New task" });
    await userEvent.type(input, "   {Enter}");
    await settle();

    expect(createBodies).toHaveLength(0);
    expect((input as HTMLInputElement).value).toBe("   ");
  });

  test("typing @ opens the assignee picker and assigns the task", async () => {
    serveBoard(BOARD);
    renderBoard();

    const backlog = await findColumn("Backlog");
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Add a task…" }),
    );
    const input = within(backlog).getByRole("textbox", { name: "New task" });
    await userEvent.type(input, "Plan offsite @");

    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Priya/ }),
    );
    expect((input as HTMLInputElement).value).toBe("Plan offsite ");
    expect(
      within(backlog).getByRole("button", { name: /Assignee: Priya/ }),
    ).toBeDefined();

    await userEvent.type(input, "{Enter}");

    await waitFor(() => expect(createBodies).toHaveLength(1));
    expect(createBodies[0]).toMatchObject({
      title: "Plan offsite",
      board_column: "backlog",
      expert_id: "expert-priya",
    });
  });

  test("Escape closes the new task input", async () => {
    serveBoard(BOARD);
    renderBoard();

    const backlog = await findColumn("Backlog");
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Add a task…" }),
    );
    await userEvent.type(
      within(backlog).getByRole("textbox", { name: "New task" }),
      "Never mind{Escape}",
    );
    await settle();

    expect(within(backlog).queryByRole("textbox", { name: "New task" })).toBe(
      null,
    );
    expect(createBodies).toHaveLength(0);
  });

  test("an empty board focuses the Backlog input", async () => {
    serveBoard({ columns: COLUMNS, sessions: [] });
    renderBoard();

    const backlog = await findColumn("Backlog");
    const input = within(backlog).getByPlaceholderText(
      "What do you want to get done?",
    );
    expect(document.activeElement).toBe(input);
    expect(
      within(backlog).getByText("Type @ to pick who works on it."),
    ).toBeDefined();
  });

  test("clicking a card opens it through ?sessionId", async () => {
    serveBoard(BOARD);
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderBoard(onUrlUpdate);

    const backlog = await findColumn("Backlog");
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Draft launch plan" }),
    );

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("sessionId")).toBe(
      "s-draft",
    );
  });

  test("the paperclip lists the card's files without opening the card", async () => {
    serveBoard(BOARD);
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          {
            id: "f-1",
            name: "release-notes.md",
            path: "/sessions/s-ship/release-notes.md",
            mime_type: "text/markdown",
            size_bytes: 120,
            origin: "generated",
            created_at: NOW,
          },
        ],
      }),
    );
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderBoard(onUrlUpdate);

    const done = await findColumn("Done");
    await userEvent.click(
      within(done).getByRole("button", { name: "2 files" }),
    );

    expect(await screen.findByText("release-notes.md")).toBeDefined();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  test("Show older asks the board for older sessions", async () => {
    serveBoard(BOARD);
    renderBoard();

    const done = await findColumn("Done");
    await userEvent.click(
      within(done).getByRole("button", { name: "Show older" }),
    );

    await waitFor(() =>
      expect(
        boardRequests.some(
          (url) => url.searchParams.get("include_older") === "true",
        ),
      ).toBe(true),
    );
  });

  test("shows an error with a retry when the board fails to load", async () => {
    server.use(
      http.get("*/api/chat/board", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    renderBoard();

    const retry = await screen.findByRole("button", { name: /try again/i });
    serveBoard(BOARD);
    await userEvent.click(retry);

    expect(await findColumn("Backlog")).toBeDefined();
  });
});
