import { getGetV2GetBoardMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
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
const flags = vi.hoisted(() => ({ hireExperts: true }));

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

vi.mock("@/app/(platform)/copilot/CopilotChatHost", () => ({
  CopilotChatHost: () => <div>chat host</div>,
}));

const NOW = new Date().toISOString();
const OLD_ID = "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b";
const NEW_ID = "8d7c6b5a-4f3e-4d2c-9b1a-0f9e8d7c6b5a";
const REASSIGN_URL = "*/api/chat/sessions/:sessionId/reassign";
const RUNNING_MESSAGE =
  "This task is still running. Wait for it to finish, then reassign.";

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

const OLD = card(OLD_ID, {
  title: "Research competitors",
  board_column: "in_progress",
  board_position: 0,
  last_line: "Found three rivals.",
});

const OLD_AFTER = {
  ...OLD,
  board_column: "done",
  board_position: -1024,
  last_line: "Reassigned to Priya",
};

const NEW = card(NEW_ID, {
  title: "Research competitors",
  board_column: "in_progress",
  board_position: -1024,
  expert_id: "expert-priya",
  is_processing: true,
});

const PRIYA = {
  id: "expert-priya",
  name: "Priya",
  avatar_url: null,
  role: "Product manager",
  is_archived: false,
};

const SAM = {
  id: "expert-sam",
  name: "Sam",
  avatar_url: null,
  role: "Analyst",
  is_archived: false,
};

function board(...sessions: SessionSummaryResponse[]): BoardResponse {
  return { columns: COLUMNS, sessions };
}

let boardRequests = 0;

function serveBoard(first: BoardResponse, later: BoardResponse = first) {
  server.use(
    getGetV2GetBoardMockHandler(() => {
      boardRequests += 1;
      return boardRequests === 1 ? first : later;
    }),
  );
}

function renderPanel(onUrlUpdate?: (event: UrlUpdateEvent) => void) {
  return render(
    <NuqsTestingAdapter
      searchParams={`?sessionId=${OLD_ID}`}
      onUrlUpdate={onUrlUpdate}
      hasMemory
    >
      <TaskBoardPage />
    </NuqsTestingAdapter>,
  );
}

function column(name: string) {
  // The open sheet is modal, so the board behind it is aria-hidden.
  return screen.getByRole("region", { name, hidden: true });
}

async function openReassign() {
  const panel = await screen.findByRole("dialog", {
    name: "Research competitors",
  });
  await userEvent.click(
    within(panel).getByRole("button", { name: /Reassign/ }),
  );
  return screen.findByRole("dialog", { name: "Reassign task" });
}

beforeEach(() => {
  boardRequests = 0;
  flags.hireExperts = true;
  toastSpy.mockReset();
  server.use(getListExpertIdentitiesMockHandler([PRIYA, SAM]));
});

afterEach(() => {
  flags.hireExperts = true;
});

describe("reassign", () => {
  test("hands the task to Priya: old card to Done, new card opens in In progress", async () => {
    const bodies: unknown[] = [];
    let releaseReassign = () => {};
    const reassignHeld = new Promise<void>((resolve) => {
      releaseReassign = resolve;
    });
    let releaseRefetch = () => {};
    const refetchHeld = new Promise<void>((resolve) => {
      releaseRefetch = resolve;
    });
    server.use(
      getGetV2GetBoardMockHandler(async () => {
        boardRequests += 1;
        if (boardRequests === 1) return board(OLD);
        // Hold the refetch: the result must come from the cache update.
        await refetchHeld;
        return board(NEW, OLD_AFTER);
      }),
      http.post(REASSIGN_URL, async ({ request }) => {
        bodies.push(await request.json());
        await reassignHeld;
        return HttpResponse.json({ old_session: OLD_AFTER, new_session: NEW });
      }),
    );
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    renderPanel(onUrlUpdate);

    const dialog = await openReassign();
    expect(within(dialog).queryByRole("radio", { name: "Otto" })).toBeNull();
    const confirm = within(dialog).getByRole("button", { name: "Reassign" });
    expect(confirm).toHaveProperty("disabled", true);

    await userEvent.click(within(dialog).getByRole("radio", { name: "Priya" }));
    expect(confirm).toHaveProperty("disabled", false);
    await userEvent.click(confirm);

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({ expert_id: "expert-priya" });
    expect(
      within(dialog).getByRole("button", { name: "Reassign" }),
    ).toHaveProperty("disabled", true);

    releaseReassign();

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Reassign task" }),
      ).toBeNull(),
    );
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("sessionId")).toBe(
      NEW_ID,
    );
    const done = within(column("Done")).getByRole("article", {
      name: "Research competitors",
      hidden: true,
    });
    expect(within(done).getByText("Reassigned to Priya")).toBeDefined();
    const inProgress = within(column("In progress")).getByRole("article", {
      name: "Research competitors",
      hidden: true,
    });
    expect(within(inProgress).getByText("Working…")).toBeDefined();
    expect(
      await screen.findByRole("dialog", { name: "Research competitors" }),
    ).toBeDefined();

    releaseRefetch();
  });

  test("offers Otto, not the current expert, for an expert's card", async () => {
    serveBoard(board({ ...OLD, expert_id: "expert-priya" }));
    renderPanel();

    const dialog = await openReassign();
    expect(within(dialog).getByRole("radio", { name: "Otto" })).toBeDefined();
    expect(within(dialog).getByRole("radio", { name: "Sam" })).toBeDefined();
    expect(within(dialog).queryByRole("radio", { name: "Priya" })).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Cancel" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Reassign task" }),
      ).toBeNull(),
    );
    expect(
      screen.getByRole("dialog", { name: "Research competitors" }),
    ).toBeDefined();
  });

  test("a running task says so and keeps the dialog open", async () => {
    serveBoard(board(OLD));
    server.use(
      http.post(REASSIGN_URL, () =>
        HttpResponse.json(
          { detail: "Reassign the task once it stops working" },
          { status: 409 },
        ),
      ),
    );
    renderPanel();

    const dialog = await openReassign();
    await userEvent.click(within(dialog).getByRole("radio", { name: "Priya" }));
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Reassign" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: RUNNING_MESSAGE }),
      ),
    );
    expect(screen.getByRole("dialog", { name: "Reassign task" })).toBeDefined();
    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Reassign" }),
      ).toHaveProperty("disabled", false),
    );
  });

  test("an expert that can't take it shows the server's reason", async () => {
    serveBoard(board(OLD));
    server.use(
      http.post(REASSIGN_URL, () =>
        HttpResponse.json({ detail: "Expert not found" }, { status: 404 }),
      ),
    );
    renderPanel();

    const dialog = await openReassign();
    await userEvent.click(within(dialog).getByRole("radio", { name: "Sam" }));
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Reassign" }),
    );

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't reassign the task",
          description: "Expert not found",
        }),
      ),
    );
    expect(screen.getByRole("dialog", { name: "Reassign task" })).toBeDefined();
  });

  test("hides Reassign when experts are off", async () => {
    flags.hireExperts = false;
    serveBoard(board(OLD));
    renderPanel();

    const panel = await screen.findByRole("dialog", {
      name: "Research competitors",
    });
    expect(within(panel).getByText("chat host")).toBeDefined();
    expect(
      within(panel).queryByRole("button", { name: /Reassign/ }),
    ).toBeNull();
  });
});
