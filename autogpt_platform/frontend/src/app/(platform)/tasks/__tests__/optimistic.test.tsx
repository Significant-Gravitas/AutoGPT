import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TaskList } from "../components/TaskList/TaskList";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TITLE = "Plan the offsite";
const REFUSED =
  "It is being worked on. Stop it first, then move it back to To do.";
let serverTasks: Task[];
let answer: ((ok: boolean) => void) | null;

/** Holds the request until the test answers it, ok or not. */
function held<T>(apply: (body: T) => void) {
  return async ({ request }: { request: Request }) => {
    const text = await request.text();
    const ok = await new Promise<boolean>((resolve) => {
      answer = resolve;
    });
    answer = null;
    if (!ok) {
      return HttpResponse.json({ detail: REFUSED }, { status: 409 });
    }
    apply(text ? (JSON.parse(text) as T) : ({} as T));
    return HttpResponse.json(serverTasks[0] ?? null);
  };
}

beforeEach(() => {
  mockToast.mockClear();
  answer = null;
  serverTasks = [makeTask("task-1", TITLE, "IN_PROGRESS")];
  server.use(
    http.get("/api/proxy/api/tasks", () =>
      HttpResponse.json({ tasks: serverTasks }),
    ),
    http.patch(
      "/api/proxy/api/tasks/task-1",
      held<Partial<Task>>((body) => {
        serverTasks = [{ ...serverTasks[0], ...body }];
      }),
    ),
    http.delete(
      "/api/proxy/api/tasks/task-1",
      held(() => {
        serverTasks = [];
      }),
    ),
  );
});

function Harness() {
  const tasks = useListTasks(
    { limit: 100 },
    { query: { select: (res) => okData(res)?.tasks ?? [] } },
  );
  return (
    <TaskList
      tasks={tasks.data ?? []}
      experts={[]}
      selectedID={null}
      onOpen={vi.fn()}
    />
  );
}

function group(name: string) {
  return screen.getByRole("region", { name: new RegExp(`^${name}`) });
}

async function openMenu() {
  await userEvent.click(
    await screen.findByRole("button", { name: `Actions for ${TITLE}` }),
  );
}

async function settle(ok: boolean) {
  await vi.waitFor(() => expect(answer).not.toBeNull());
  answer?.(ok);
}

describe("Task changes show at once", () => {
  it("shows a new title before the server answers, and keeps it", async () => {
    render(<Harness />);
    await openMenu();
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Rename" }),
    );
    const field = await screen.findByLabelText("Task title");
    await userEvent.clear(field);
    await userEvent.type(field, "Plan the retreat{Enter}");

    expect(await screen.findByText("Plan the retreat")).toBeDefined();
    expect(screen.queryByText(TITLE)).toBeNull();

    await settle(true);
    await vi.waitFor(() => expect(answer).toBeNull());
    expect(await screen.findByText("Plan the retreat")).toBeDefined();
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("moves a task to Done at once, and back with a toast when refused", async () => {
    render(<Harness />);
    await openMenu();
    const status = await screen.findByRole("menuitem", {
      name: "Change status",
    });
    status.focus();
    await userEvent.keyboard("{ArrowRight}");
    const done = await screen.findByRole("menuitemradio", { name: "Done" });
    done.focus();
    await userEvent.keyboard("{Enter}");

    expect(within(group("Done")).getByText(TITLE)).toBeDefined();

    await settle(false);

    const working = await screen.findByRole("region", { name: /^Working/ });
    expect(within(working).getByText(TITLE)).toBeDefined();
    expect(screen.queryByRole("region", { name: /^Done/ })).toBeNull();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't change the status",
        description: REFUSED,
        variant: "destructive",
      }),
    );
  });

  it("takes an archived task off at once, and puts it back when refused", async () => {
    render(<Harness />);
    await openMenu();
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Archive" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Archive task" }),
    );

    await vi.waitFor(() =>
      expect(screen.queryByRole("button", { name: /^Plan the offsite/ })).toBe(
        null,
      ),
    );

    await settle(false);

    expect(
      await screen.findByRole("button", { name: /^Plan the offsite/ }),
    ).toBeDefined();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't archive the task" }),
    );
  });
});
