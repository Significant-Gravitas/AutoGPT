import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useWorkAreaStore } from "../../../workArea";
import { HomeBoard } from "../components/HomeBoard/HomeBoard";
import { expectTypingKept } from "@/tests/integrations/typing";
import { makeTask } from "./taskFixtures";

const PAGE_ONE: Task[] = [
  makeTask("t1", "Pick a venue for the offsite", "NEEDS_INPUT"),
  makeTask("t2", "Draft the launch email", "IN_PROGRESS", "maria"),
];
const PAGE_TWO: Task[] = [makeTask("t3", "Send the invoice", "DONE")];

let requests: URLSearchParams[];
let onUrlUpdate: ReturnType<typeof vi.fn<(event: UrlUpdateEvent) => void>>;
let answerPatch: ((ok: boolean) => void) | null;

function page(tasks: Task[], next: number | null) {
  return HttpResponse.json({
    tasks,
    has_more: next !== null,
    next_offset: next,
  });
}

beforeEach(() => {
  requests = [];
  answerPatch = null;
  onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
  useWorkAreaStore.getState().reset();
  server.use(
    getListExpertsMockHandler200([]),
    http.get("/api/proxy/api/tasks", ({ request }) => {
      const params = new URL(request.url).searchParams;
      requests.push(params);
      const q = params.get("q");
      if (q) {
        return page(
          [...PAGE_ONE, ...PAGE_TWO].filter((task) =>
            task.title.toLowerCase().includes(q.toLowerCase()),
          ),
          null,
        );
      }
      if (params.get("owned_by_otto") === "true") {
        return page([PAGE_ONE[0], ...PAGE_TWO], null);
      }
      if (params.get("statuses") === "DONE") return page(PAGE_TWO, null);
      return params.get("offset") === "50"
        ? page(PAGE_TWO, null)
        : page(PAGE_ONE, 50);
    }),
    http.patch("/api/proxy/api/tasks/t3", async () => {
      const ok = await new Promise<boolean>((resolve) => {
        answerPatch = resolve;
      });
      return ok
        ? HttpResponse.json({ ...PAGE_TWO[0], title: "Send the bill" })
        : HttpResponse.json({ detail: "No." }, { status: 500 });
    }),
  );
});

function renderBoard(searchParams = "?show=tasks") {
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
    >
      <HomeBoard />
    </NuqsTestingAdapter>,
  );
}

function titles() {
  return screen
    .queryAllByTestId("board-card")
    .map((card) => within(card).getAllByRole("button")[0].id);
}

describe("Home Tasks — search, filters and more pages", () => {
  it("searches by title once typing pauses, keeps it in the address, and resets", async () => {
    const user = userEvent.setup();
    renderBoard();
    await screen.findAllByTestId("board-card");

    await user.type(
      screen.getByRole("searchbox", { name: "Search tasks" }),
      "invoice",
    );

    await vi.waitFor(() => expect(requests.at(-1)?.get("q")).toBe("invoice"));
    await vi.waitFor(() => expect(titles()).toEqual(["task-row-t3"]));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("q")).toBe(
      "invoice",
    );
    expect(requests.filter((params) => params.get("q") === "i")).toHaveLength(
      0,
    );

    await user.clear(screen.getByRole("searchbox", { name: "Search tasks" }));
    await user.type(
      screen.getByRole("searchbox", { name: "Search tasks" }),
      "zzz",
    );
    expect(await screen.findByText("No tasks match")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Reset filters" }));
    await vi.waitFor(() => expect(titles()).toHaveLength(2));
    expect(
      (
        screen.getByRole("searchbox", {
          name: "Search tasks",
        }) as HTMLInputElement
      ).value,
    ).toBe("");
  });

  it("keeps every keystroke in the search box while the address follows", async () => {
    renderBoard();
    await screen.findAllByTestId("board-card");

    await expectTypingKept(
      screen.getByRole("searchbox", {
        name: "Search tasks",
      }) as HTMLInputElement,
      "venue 日本",
    );
  });

  it("narrows by owner and by status", async () => {
    const user = userEvent.setup();
    renderBoard();
    await screen.findAllByTestId("board-card");

    await user.click(
      screen.getByRole("button", { name: "Owner: Anyone. Change owner" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Otto" }),
    );
    await vi.waitFor(() =>
      expect(requests.at(-1)?.get("owned_by_otto")).toBe("true"),
    );
    await vi.waitFor(() =>
      expect(titles()).toEqual(["task-row-t1", "task-row-t3"]),
    );

    await user.click(screen.getByRole("button", { name: "Reset" }));
    await user.click(
      await screen.findByRole("button", {
        name: "Status: Any status. Change status",
      }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Done" }),
    );
    await vi.waitFor(() =>
      expect(requests.at(-1)?.getAll("statuses")).toEqual(["DONE"]),
    );
    await vi.waitFor(() => expect(titles()).toEqual(["task-row-t3"]));
  });

  it("reads more with Show more, and patches a task on a later page at once", async () => {
    const user = userEvent.setup();
    renderBoard();
    await screen.findAllByTestId("board-card");
    expect(titles()).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Show more tasks" }));

    await vi.waitFor(() => expect(titles()).toHaveLength(3));
    expect(requests.at(-1)?.get("offset")).toBe("50");
    expect(
      screen.queryByRole("button", { name: "Show more tasks" }),
    ).toBeNull();

    await user.click(
      screen.getByRole("button", { name: "Actions for Send the invoice" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Rename" }));
    const field = await screen.findByLabelText("Task title");
    await user.clear(field);
    await user.type(field, "Send the bill{Enter}");

    expect(
      await screen.findByRole("button", { name: /^Send the bill/ }),
    ).toBeDefined();
    await vi.waitFor(() => expect(answerPatch).not.toBeNull());
    answerPatch?.(true);
  });
});
