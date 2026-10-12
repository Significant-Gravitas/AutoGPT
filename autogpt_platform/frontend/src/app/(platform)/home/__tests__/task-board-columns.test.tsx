import {
  getGetV2GetBoardMockHandler,
  getPutV2UpdateBoardColumnsMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
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
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, test, vi } from "vitest";
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

const NOW = new Date().toISOString();
const REVIEW_KEY = "col_4b9e2f0a-1c3d-4e5f-8a7b-6c5d4e3f2a1b";

const BUILTINS: BoardColumn[] = [
  { key: "backlog", name: "Backlog", builtin: true },
  { key: "in_progress", name: "In progress", builtin: true },
  { key: "done", name: "Done", builtin: true },
];

const REVIEW: BoardColumn = { key: REVIEW_KEY, name: "Review", builtin: false };

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

const SESSIONS = [
  card("s-draft", {
    title: "Draft launch plan",
    board_column: "backlog",
    board_position: 0,
    started: false,
  }),
  card("s-check", {
    title: "Check the copy",
    board_column: REVIEW_KEY,
    board_position: 0,
  }),
];

let stored: BoardResponse;
let putBodies: { columns: BoardColumn[] }[] = [];

function serveBoard(columns: BoardColumn[]) {
  stored = { columns, sessions: SESSIONS };
  server.use(
    getGetV2GetBoardMockHandler(() => stored),
    getPutV2UpdateBoardColumnsMockHandler(async ({ request }) => {
      const body = (await request.json()) as { columns: BoardColumn[] };
      putBodies.push(body);
      stored = { ...stored, columns: body.columns };
      return body;
    }),
  );
}

function renderBoard() {
  return render(
    <NuqsTestingAdapter>
      <TaskBoardPage />
    </NuqsTestingAdapter>,
  );
}

async function openMenu(column: string) {
  const region = await screen.findByRole("region", { name: column });
  await userEvent.click(
    within(region).getByRole("button", { name: `${column} column actions` }),
  );
  return screen.findByRole("menu");
}

beforeEach(() => {
  putBodies = [];
  toastSpy.mockReset();
  server.use(getListExpertIdentitiesMockHandler([]));
});

describe("board columns", () => {
  test("renames Backlog and keeps it built in", async () => {
    serveBoard([...BUILTINS, REVIEW]);
    renderBoard();

    const menu = await openMenu("Backlog");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Column name" });
    await waitFor(() => expect(document.activeElement).toBe(input));
    await userEvent.clear(input);
    await userEvent.type(input, "  Ideas  {Enter}");

    await waitFor(() => expect(putBodies).toHaveLength(1));
    expect(putBodies[0].columns[0]).toEqual({
      key: "backlog",
      name: "Ideas",
      builtin: true,
    });
    expect(putBodies[0].columns).toHaveLength(4);
    const ideas = await screen.findByRole("region", { name: "Ideas" });
    await waitFor(() =>
      expect(document.activeElement).toBe(
        within(ideas).getByRole("button", { name: "Ideas column actions" }),
      ),
    );
  });

  test("Escape in the name field gives focus back to the column's actions", async () => {
    serveBoard(BUILTINS);
    renderBoard();

    const menu = await openMenu("Done");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Column name" });
    await waitFor(() => expect(document.activeElement).toBe(input));
    await userEvent.type(input, "{Escape}");

    const done = screen.getByRole("region", { name: "Done" });
    await waitFor(() =>
      expect(document.activeElement).toBe(
        within(done).getByRole("button", { name: "Done column actions" }),
      ),
    );
    expect(putBodies).toHaveLength(0);
  });

  test("does not save a blank name", async () => {
    serveBoard(BUILTINS);
    renderBoard();

    const menu = await openMenu("Backlog");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    );
    const input = await screen.findByRole("textbox", { name: "Column name" });
    await waitFor(() => expect(document.activeElement).toBe(input));
    await userEvent.clear(input);
    await userEvent.type(input, "   {Enter}");

    expect(await screen.findByText("Give the column a name.")).toBeDefined();
    expect(screen.getByRole("textbox", { name: "Column name" })).toBeDefined();
    expect(putBodies).toHaveLength(0);

    await userEvent.type(input, "{Escape}");
    expect(screen.queryByRole("textbox", { name: "Column name" })).toBeNull();
    expect(screen.getByRole("region", { name: "Backlog" })).toBeDefined();
    expect(putBodies).toHaveLength(0);
  });

  test("adds a column at the end", async () => {
    serveBoard(BUILTINS);
    renderBoard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Add column" }),
    );

    await waitFor(() => expect(putBodies).toHaveLength(1));
    const columns = putBodies[0].columns;
    expect(columns).toHaveLength(4);
    expect(columns.slice(0, 3)).toEqual(BUILTINS);
    expect(columns[3].key).toMatch(/^col_/);
    expect(columns[3].name).toBe("New column");
    expect(
      await screen.findByRole("region", { name: "New column" }),
    ).toBeDefined();
  });

  test("adds a column after another one", async () => {
    serveBoard(BUILTINS);
    renderBoard();

    const menu = await openMenu("Backlog");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Add column after" }),
    );

    await waitFor(() => expect(putBodies).toHaveLength(1));
    const keys = putBodies[0].columns.map((column) => column.key);
    expect(keys).toHaveLength(4);
    expect(keys[0]).toBe("backlog");
    expect(keys[1]).toMatch(/^col_/);
    expect(keys.slice(2)).toEqual(["in_progress", "done"]);
  });

  test("deletes a custom column after confirming and moves its cards to Backlog", async () => {
    serveBoard([...BUILTINS, REVIEW]);
    renderBoard();

    const menu = await openMenu("Review");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Delete" }),
    );

    const dialog = await screen.findByRole("dialog", {
      name: "Delete “Review”?",
    });
    expect(
      within(dialog).getByText("Its cards move to Backlog."),
    ).toBeDefined();
    expect(putBodies).toHaveLength(0);
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Delete" }),
    );

    await waitFor(() => expect(putBodies).toHaveLength(1));
    expect(putBodies[0].columns).toEqual(BUILTINS);
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Review" })).toBeNull(),
    );
    const backlog = screen.getByRole("region", { name: "Backlog" });
    expect(
      within(backlog).getByRole("article", { name: "Check the copy" }),
    ).toBeDefined();
    await waitFor(() =>
      expect(document.activeElement).toBe(
        within(backlog).getByRole("button", { name: "Backlog column actions" }),
      ),
    );
  });

  test("cancelling the delete keeps the column", async () => {
    serveBoard([...BUILTINS, REVIEW]);
    renderBoard();

    const menu = await openMenu("Review");
    await userEvent.click(
      within(menu).getByRole("menuitem", { name: "Delete" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Delete “Review”?",
    });
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Cancel" }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const review = screen.getByRole("region", { name: "Review" });
    await waitFor(() =>
      expect(document.activeElement).toBe(
        within(review).getByRole("button", { name: "Review column actions" }),
      ),
    );
    expect(putBodies).toHaveLength(0);
  });

  test("built-in columns cannot be deleted", async () => {
    serveBoard(BUILTINS);
    renderBoard();

    for (const name of ["Backlog", "In progress", "Done"]) {
      const menu = await openMenu(name);
      expect(
        within(menu).getByRole("menuitem", { name: "Rename" }),
      ).toBeDefined();
      expect(
        within(menu).queryByRole("menuitem", { name: "Delete" }),
      ).toBeNull();
      await userEvent.keyboard("{Escape}");
      await waitFor(() => expect(screen.queryByRole("menu")).toBeNull());
    }
  });

  test("hides Add column at six columns", async () => {
    const extra = [1, 2, 3].map((n) => ({
      key: `col_00000000-0000-4000-8000-00000000000${n}`,
      name: `Extra ${n}`,
      builtin: false,
    }));
    serveBoard([...BUILTINS, ...extra]);
    renderBoard();

    const menu = await openMenu("Extra 3");
    expect(
      within(menu).getByRole("menuitem", { name: "Rename" }),
    ).toBeDefined();
    expect(
      within(menu).queryByRole("menuitem", { name: "Add column after" }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Add column" })).toBeNull();
  });

  test("a failed save puts the columns back and says so", async () => {
    serveBoard(BUILTINS);
    server.use(
      http.put("*/api/chat/board/columns", () =>
        HttpResponse.json({ detail: "boom" }, { status: 422 }),
      ),
    );
    renderBoard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Add column" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't save the columns" }),
      ),
    );
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "New column" })).toBeNull(),
    );
    expect(screen.getAllByRole("region")).toHaveLength(3);
  });
});
