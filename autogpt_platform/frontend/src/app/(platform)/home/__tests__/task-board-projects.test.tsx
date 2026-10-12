import {
  getDeleteV2DeleteBoardMockHandler,
  getGetV2GetBoardMockHandler,
  getGetV2GetSessionMockHandler200,
  getPatchV2UpdateBoardMockHandler,
  getPatchV2UpdateSessionBoardMockHandler,
  getPostV2CreateSessionMockHandler,
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
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { BOARDS_LIST_QUERY_KEY } from "@/components/layout/AppSidebar/components/ProjectsNavSection/helpers";
import { boardQueryKey } from "../board-helpers";
import { TaskBoardPage } from "../TaskBoardPage";

const { routerReplace } = vi.hoisted(() => ({ routerReplace: vi.fn() }));

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
    useRouter: () => ({
      push: vi.fn(),
      replace: routerReplace,
      prefetch: vi.fn(),
    }),
    usePathname: () => "/home",
    useSearchParams: () => new URLSearchParams(),
  };
});

vi.mock("@/app/(platform)/copilot/CopilotChatHost", () => ({
  CopilotChatHost: () => <div>chat host</div>,
}));

const NOW = new Date().toISOString();
const B1 = "b1a2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d";
const S_HOME = "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b";
const S_B1 = "8d7c6b5a-4f3e-4d2c-9b1a-0f9e8d7c6b5a";
const S9 = "9e9e9e9e-1a2b-4c3d-8e4f-5a6b7c8d9e0f";

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

const HOME: BoardResponse = {
  board: { id: "home-id", name: "Home", is_default: true, position: 0 },
  columns: COLUMNS,
  sessions: [
    card(S_HOME, {
      title: "Plan offsite",
      board_column: "backlog",
      board_position: 0,
    }),
  ],
};

const PROJECT: BoardResponse = {
  board: { id: B1, name: "Launch", is_default: false, position: 1 },
  columns: COLUMNS,
  sessions: [
    card(S_B1, {
      title: "Write launch post",
      board_column: "in_progress",
      board_position: 0,
      board_id: B1,
    }),
  ],
};

let boardPaths: string[] = [];

function serveBoards(boards: Record<string, BoardResponse>) {
  server.use(
    getGetV2GetBoardMockHandler(({ request, params }) => {
      boardPaths.push(new URL(request.url).pathname);
      const board = boards[String(params.boardId)];
      if (!board) throw HttpResponse.json(null, { status: 404 });
      return board;
    }),
  );
}

function renderBoard(
  boardRef: string,
  search = "",
  client?: QueryClient,
  onUrlUpdate?: (event: UrlUpdateEvent) => void,
) {
  const page = (
    <NuqsTestingAdapter
      searchParams={search}
      hasMemory
      onUrlUpdate={onUrlUpdate}
    >
      <TaskBoardPage boardRef={boardRef} />
    </NuqsTestingAdapter>
  );
  return render(
    client ? (
      <QueryClientProvider client={client}>{page}</QueryClientProvider>
    ) : (
      page
    ),
  );
}

async function openProjectMenu() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Project actions" }),
  );
  return screen.findByRole("menu");
}

beforeEach(() => {
  boardPaths = [];
  routerReplace.mockClear();
  server.use(getListExpertIdentitiesMockHandler([]));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("task board projects", () => {
  test("a project shows its own name and cards", async () => {
    serveBoards({ default: HOME, [B1]: PROJECT });
    renderBoard(B1);

    expect(
      await screen.findByRole("heading", { name: "Launch" }),
    ).toBeDefined();
    const inProgress = await screen.findByRole("region", {
      name: "In progress",
    });
    expect(
      within(inProgress).getByRole("article", { name: "Write launch post" }),
    ).toBeDefined();
    expect(screen.queryByRole("article", { name: "Plan offsite" })).toBeNull();
    expect(boardPaths.every((path) => path.endsWith(`/boards/${B1}`))).toBe(
      true,
    );
    expect(screen.queryByRole("button", { name: "Show older" })).toBeNull();
  });

  test("a task added on a project goes to that project", async () => {
    const bodies: Record<string, unknown>[] = [];
    serveBoards({ [B1]: PROJECT });
    server.use(
      getPostV2CreateSessionMockHandler(async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>);
        return { id: "s-new", created_at: NOW, user_id: "user-1" };
      }),
    );
    renderBoard(B1);

    const backlog = await screen.findByRole("region", { name: "Backlog" });
    await userEvent.click(
      within(backlog).getByRole("button", { name: "Add a task…" }),
    );
    await userEvent.type(
      within(backlog).getByRole("textbox", { name: "New task" }),
      "Draft the email{Enter}",
    );

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toMatchObject({
      title: "Draft the email",
      board_column: "backlog",
      board_id: B1,
    });
  });

  test("a move on a project leaves the cached Home board alone", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const homeKey = boardQueryKey("default", { include_older: false });
    // A stale Home entry still holding the card, as after a move to B1.
    const staleCard = { ...PROJECT.sessions[0], board_column: "backlog" };
    client.setQueryData(homeKey, {
      status: 200,
      data: { ...HOME, sessions: [...HOME.sessions, staleCard] },
      headers: new Headers(),
    });
    const cachedHome = client.getQueryData(homeKey);
    const moves: unknown[] = [];
    serveBoards({ [B1]: PROJECT });
    server.use(
      getPatchV2UpdateSessionBoardMockHandler(async ({ request }) => {
        moves.push(await request.json());
        return { ...PROJECT.sessions[0], board_column: "done" };
      }),
    );
    renderBoard(B1, `?sessionId=${S_B1}`, client);

    const dialog = await screen.findByRole("dialog", {
      name: "Write launch post",
    });
    await userEvent.click(
      within(dialog).getByRole("combobox", { name: "Column" }),
    );
    await userEvent.click(await screen.findByRole("option", { name: "Done" }));

    await waitFor(() => expect(moves).toEqual([{ column: "done" }]));
    await waitFor(() => expect(boardPaths.length).toBeGreaterThan(1));
    expect(client.getQueryData(homeKey)).toBe(cachedHome);
    expect(client.getQueryState(homeKey)?.isInvalidated).toBe(false);
  });

  test("renaming a project saves and shows the new name", async () => {
    const renames: unknown[] = [];
    let current = PROJECT;
    server.use(
      getGetV2GetBoardMockHandler(() => current),
      getPatchV2UpdateBoardMockHandler(async ({ request, params }) => {
        const body = (await request.json()) as { name: string };
        renames.push({ boardId: params.boardId, ...body });
        current = { ...PROJECT, board: { ...PROJECT.board, name: body.name } };
        return current.board;
      }),
    );
    renderBoard(B1);

    const menu = await openProjectMenu();
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Project name" });
    await userEvent.clear(input);
    await userEvent.type(input, "  Launch v2  {Enter}");

    await waitFor(() =>
      expect(renames).toEqual([{ boardId: B1, name: "Launch v2" }]),
    );
    expect(
      await screen.findByRole("heading", { name: "Launch v2" }),
    ).toBeDefined();
  });

  test("Escape leaves the name as it was", async () => {
    const renames: unknown[] = [];
    serveBoards({ [B1]: PROJECT });
    server.use(
      getPatchV2UpdateBoardMockHandler(async ({ request }) => {
        renames.push(await request.json());
        return PROJECT.board;
      }),
    );
    renderBoard(B1);

    const menu = await openProjectMenu();
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Project name" });
    await userEvent.clear(input);
    await userEvent.type(input, "Other{Escape}");

    expect(screen.getByRole("heading", { name: "Launch" })).toBeDefined();
    expect(renames).toEqual([]);
  });

  test("a blank name is refused", async () => {
    serveBoards({ [B1]: PROJECT });
    renderBoard(B1);

    const menu = await openProjectMenu();
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Project name" });
    await userEvent.clear(input);
    await userEvent.type(input, "   {Enter}");

    expect(await screen.findByText("Give the project a name.")).toBeDefined();
  });

  test("a new project opens with its name ready to type over", async () => {
    const renames: unknown[] = [];
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    client.setQueryData(BOARDS_LIST_QUERY_KEY, {
      status: 200,
      data: { boards: [] },
      headers: new Headers(),
    });
    serveBoards({ [B1]: PROJECT });
    server.use(
      getPatchV2UpdateBoardMockHandler(async ({ request }) => {
        renames.push(await request.json());
        return PROJECT.board;
      }),
    );
    renderBoard(B1, "?rename=1", client, onUrlUpdate);

    const input = await screen.findByRole("textbox", { name: "Project name" });
    expect((input as HTMLInputElement).value).toBe("Launch");
    await waitFor(() => expect(document.activeElement).toBe(input));
    await userEvent.keyboard("Q3 launch{Enter}");

    await waitFor(() => expect(renames).toEqual([{ name: "Q3 launch" }]));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.has("rename")).toBe(
      false,
    );
    await waitFor(() =>
      expect(client.getQueryState(BOARDS_LIST_QUERY_KEY)?.isInvalidated).toBe(
        true,
      ),
    );
  });

  test("Home ignores ?rename=1", async () => {
    serveBoards({ default: HOME });
    renderBoard("default", "?rename=1");

    expect(await screen.findByRole("heading", { name: "Home" })).toBeDefined();
    expect(screen.queryByRole("textbox", { name: "Project name" })).toBeNull();
  });

  test("deleting a project confirms, deletes it and goes Home", async () => {
    const deleted: string[] = [];
    serveBoards({ [B1]: PROJECT });
    server.use(
      getDeleteV2DeleteBoardMockHandler(({ params }) => {
        deleted.push(String(params.boardId));
      }),
    );
    renderBoard(B1);

    const menu = await openProjectMenu();
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Delete" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Delete “Launch”?",
    });
    expect(
      within(dialog).getByText("Its tasks move to Home's Backlog."),
    ).toBeDefined();
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Delete" }),
    );

    await waitFor(() => expect(deleted).toEqual([B1]));
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/home"));
  });

  test("Home has no project menu", async () => {
    serveBoards({ default: HOME });
    renderBoard("default");

    expect(await screen.findByRole("heading", { name: "Home" })).toBeDefined();
    await screen.findByRole("article", { name: "Plan offsite" });
    expect(
      screen.queryByRole("button", { name: "Project actions" }),
    ).toBeNull();
    expect(screen.getByRole("button", { name: "Show older" })).toBeDefined();
    expect(boardPaths.every((path) => path.endsWith("/boards/default"))).toBe(
      true,
    );
  });

  test("a link to a session on a project goes to that project", async () => {
    serveBoards({ default: HOME, [B1]: PROJECT });
    server.use(
      getGetV2GetSessionMockHandler200({
        id: S9,
        created_at: NOW,
        updated_at: NOW,
        user_id: "user-1",
        title: "Pick a venue",
        messages: [],
        board_id: B1,
      }),
    );
    renderBoard("default", `?sessionId=${S9}`);

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith(
        `/boards/${B1}?sessionId=${S9}`,
      ),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  test("a link to a Home session from a project goes Home", async () => {
    serveBoards({ [B1]: PROJECT });
    server.use(
      getGetV2GetSessionMockHandler200({
        id: S9,
        created_at: NOW,
        updated_at: NOW,
        user_id: "user-1",
        title: "Pick a venue",
        messages: [],
        board_id: null,
      }),
    );
    renderBoard(B1, `?sessionId=${S9}`);

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith(`/home?sessionId=${S9}`),
    );
  });

  test("an older Home session that is not on the board opens on Home", async () => {
    serveBoards({ default: HOME });
    server.use(
      getGetV2GetSessionMockHandler200({
        id: S9,
        created_at: NOW,
        updated_at: NOW,
        user_id: "user-1",
        title: "Pick a venue",
        messages: [],
        board_id: null,
      }),
    );
    renderBoard("default", `?sessionId=${S9}`);

    expect(
      await screen.findByRole("dialog", { name: "Pick a venue" }),
    ).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("a project deleted elsewhere drops its cards on the next poll", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let requests = 0;
    server.use(
      http.get("*/api/chat/boards/:boardId", () => {
        requests += 1;
        return requests === 1
          ? HttpResponse.json(PROJECT)
          : HttpResponse.json({ detail: "Board not found" }, { status: 404 });
      }),
    );
    renderBoard(B1);

    expect(
      await screen.findByRole("article", { name: "Write launch post" }),
    ).toBeDefined();
    await vi.advanceTimersByTimeAsync(11_000);

    expect(await screen.findByText("This project doesn't exist")).toBeDefined();
    expect(screen.queryAllByRole("region")).toHaveLength(0);
    expect(screen.queryByRole("article")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Launch" })).toBeNull();
  });

  test("an unknown project says it doesn't exist", async () => {
    server.use(
      http.get("*/api/chat/boards/:boardId", () =>
        HttpResponse.json({ detail: "Board not found" }, { status: 404 }),
      ),
    );
    renderBoard(B1);

    expect(await screen.findByText("This project doesn't exist")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Go to Home" }).getAttribute("href"),
    ).toBe("/home");
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
  });
});
