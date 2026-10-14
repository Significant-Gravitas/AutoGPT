import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetV2GetBoardMockHandler,
  getGetV2ListBoardsMockHandler,
  getPatchV2UpdateSessionBoardMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { BoardResponse } from "@/app/api/__generated__/models/boardResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { ToastProps } from "@/components/molecules/Toast/use-toast";
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
import { beforeEach, describe, expect, test, vi } from "vitest";
import { boardQueryKey } from "../board-helpers";
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
const HOME_ID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d";
const B1 = "b1a2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d";
const B2 = "c2b3a4d5-6e7f-4b8c-9d0e-1f2a3b4c5d6e";
const S_HOME = "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b";
const S_B1 = "8d7c6b5a-4f3e-4d2c-9b1a-0f9e8d7c6b5a";

const COLUMNS: BoardResponse["columns"] = [
  { key: "backlog", name: "Backlog", builtin: true },
  { key: "in_progress", name: "In progress", builtin: true },
  { key: "done", name: "Done", builtin: true },
];

const HOME_SUMMARY = {
  id: HOME_ID,
  name: "Home",
  is_default: true,
  position: 0,
};
const LAUNCH = { id: B1, name: "Launch", is_default: false, position: 1 };
const HIRING = { id: B2, name: "Hiring", is_default: false, position: 2 };

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

const HOME_CARD = card(S_HOME, {
  title: "Plan offsite",
  board_column: "backlog",
  board_position: 0,
});
const PROJECT_CARD = card(S_B1, {
  title: "Write launch post",
  board_column: "in_progress",
  board_position: 0,
  board_id: B1,
});

let boards: Record<string, BoardResponse> = {};
let moves: { sessionId: string; body: unknown }[] = [];

function serve({ failMove = false } = {}) {
  boards = {
    default: { board: HOME_SUMMARY, columns: COLUMNS, sessions: [HOME_CARD] },
    [B1]: { board: LAUNCH, columns: COLUMNS, sessions: [PROJECT_CARD] },
    [B2]: { board: HIRING, columns: COLUMNS, sessions: [] },
  };
  server.use(
    getGetV2ListBoardsMockHandler({ boards: [HOME_SUMMARY, LAUNCH, HIRING] }),
    getGetV2GetBoardMockHandler(({ params }) => {
      const board = boards[String(params.boardId)];
      if (!board) throw HttpResponse.json(null, { status: 404 });
      return board;
    }),
    failMove
      ? http.patch("*/api/chat/sessions/:sessionId/board", () =>
          HttpResponse.json({ detail: "boom" }, { status: 500 }),
        )
      : getPatchV2UpdateSessionBoardMockHandler(async ({ request, params }) => {
          const sessionId = String(params.sessionId);
          const body = (await request.json()) as { board_id: string };
          moves.push({ sessionId, body });
          const moved = Object.values(boards)
            .flatMap((board) => board.sessions)
            .find((item) => item.id === sessionId) as SessionSummaryResponse;
          for (const board of Object.values(boards)) {
            board.sessions = board.sessions.filter(
              (item) => item.id !== sessionId,
            );
          }
          const target = boards[body.board_id];
          const result = {
            ...moved,
            board_id: body.board_id === "default" ? null : body.board_id,
          };
          target.sessions = [result, ...target.sessions];
          return result;
        }),
  );
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

function renderBoard(
  boardRef: string,
  {
    search = "",
    client = newClient(),
    onUrlUpdate,
  }: {
    search?: string;
    client?: QueryClient;
    onUrlUpdate?: (event: UrlUpdateEvent) => void;
  } = {},
) {
  render(
    <QueryClientProvider client={client}>
      <NuqsTestingAdapter
        searchParams={search}
        hasMemory
        onUrlUpdate={onUrlUpdate}
      >
        <TaskBoardPage boardRef={boardRef} />
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );
  return client;
}

async function openMoveMenu(scope: HTMLElement) {
  await userEvent.click(
    within(scope).getByRole("button", { name: "Task actions" }),
  );
  const menu = await screen.findByRole("menu");
  await userEvent.click(
    within(menu).getByRole("menuitem", { name: "Move to project" }),
  );
  const menus = await screen.findAllByRole("menu");
  return menus[menus.length - 1];
}

function lastToast() {
  return toastSpy.mock.lastCall?.[0] as ToastProps | undefined;
}

beforeEach(() => {
  moves = [];
  toastSpy.mockReset();
});

describe("move a card to another project", () => {
  test("from the card menu, the card leaves Home and the toast links to it", async () => {
    serve();
    const client = newClient();
    const projectKey = boardQueryKey(B1, { include_older: false });
    const sessionKey = getGetV2GetSessionQueryKey(S_HOME);
    const cached = { status: 200, data: {}, headers: new Headers() };
    client.setQueryData(projectKey, cached);
    client.setQueryData(sessionKey, cached);
    renderBoard("default", { client });

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    const submenu = await openMoveMenu(article);
    await userEvent.click(
      await within(submenu).findByRole("menuitem", { name: "Launch" }),
    );

    await waitFor(() =>
      expect(moves).toEqual([{ sessionId: S_HOME, body: { board_id: B1 } }]),
    );
    await waitFor(() =>
      expect(screen.queryByRole("article", { name: "Plan offsite" })).toBe(
        null,
      ),
    );
    await waitFor(() => expect(lastToast()?.title).toBe("Moved to Launch"));
    render(<>{lastToast()?.action}</>);
    expect(
      screen.getByRole("link", { name: "View" }).getAttribute("href"),
    ).toBe(`/boards/${B1}?sessionId=${S_HOME}`);
    expect(client.getQueryState(projectKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(sessionKey)?.isInvalidated).toBe(true);
  });

  test("from the panel, the move closes the panel", async () => {
    serve();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderBoard("default", { search: `?sessionId=${S_HOME}`, onUrlUpdate });

    const dialog = await screen.findByRole("dialog", { name: "Plan offsite" });
    const submenu = await openMoveMenu(dialog);
    await userEvent.click(
      await within(submenu).findByRole("menuitem", { name: "Hiring" }),
    );

    await waitFor(() =>
      expect(moves).toEqual([{ sessionId: S_HOME, body: { board_id: B2 } }]),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.has("sessionId")).toBe(
      false,
    );
    expect(screen.queryByRole("article", { name: "Plan offsite" })).toBeNull();
    await waitFor(() => expect(lastToast()?.title).toBe("Moved to Hiring"));
    render(<>{lastToast()?.action}</>);
    expect(
      screen.getByRole("link", { name: "View" }).getAttribute("href"),
    ).toBe(`/boards/${B2}?sessionId=${S_HOME}`);
  });

  test("a project offers Home and the other projects, not itself", async () => {
    serve();
    renderBoard(B1);

    const article = await screen.findByRole("article", {
      name: "Write launch post",
    });
    const submenu = await openMoveMenu(article);
    await within(submenu).findByRole("menuitem", { name: "Hiring" });
    expect(
      within(submenu)
        .getAllByRole("menuitem")
        .map((item) => item.textContent),
    ).toEqual(["Home", "Hiring"]);

    await userEvent.click(
      within(submenu).getByRole("menuitem", { name: "Home" }),
    );

    await waitFor(() =>
      expect(moves).toEqual([
        { sessionId: S_B1, body: { board_id: "default" } },
      ]),
    );
    await waitFor(() => expect(lastToast()?.title).toBe("Moved to Home"));
    render(<>{lastToast()?.action}</>);
    expect(
      screen.getByRole("link", { name: "View" }).getAttribute("href"),
    ).toBe(`/home?sessionId=${S_B1}`);
  });

  test("Home does not offer itself", async () => {
    serve();
    renderBoard("default");

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    const submenu = await openMoveMenu(article);
    await within(submenu).findByRole("menuitem", { name: "Launch" });
    expect(
      within(submenu)
        .getAllByRole("menuitem")
        .map((item) => item.textContent),
    ).toEqual(["Launch", "Hiring"]);
  });

  test("clicking the card's menu button opens the menu, not the panel", async () => {
    serve();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderBoard("default", { onUrlUpdate });

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    await userEvent.click(
      within(article).getByRole("button", { name: "Task actions" }),
    );

    expect(await screen.findByRole("menu")).toBeDefined();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  test("Enter on the card's menu button opens the menu, not the panel", async () => {
    serve();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderBoard("default", { onUrlUpdate });

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    within(article).getByRole("button", { name: "Task actions" }).focus();
    await userEvent.keyboard("{Enter}");

    expect(await screen.findByRole("menu")).toBeDefined();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  test("a project list that fails to load says so", async () => {
    serve();
    server.use(
      http.get("*/api/chat/boards", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    renderBoard("default");

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    const submenu = await openMoveMenu(article);

    expect(
      await within(submenu).findByText("Couldn't load projects"),
    ).toBeDefined();
    expect(within(submenu).queryByText("No projects yet")).toBeNull();
  });

  test("a failed move puts the card back", async () => {
    serve({ failMove: true });
    renderBoard("default");

    const article = await screen.findByRole("article", {
      name: "Plan offsite",
    });
    const submenu = await openMoveMenu(article);
    await userEvent.click(
      await within(submenu).findByRole("menuitem", { name: "Launch" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't move the task",
          variant: "destructive",
        }),
      ),
    );
    expect(
      await screen.findByRole("article", { name: "Plan offsite" }),
    ).toBeDefined();
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Moved to Launch" }),
    );
  });
});
