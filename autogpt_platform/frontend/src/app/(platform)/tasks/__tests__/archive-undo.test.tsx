import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
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
const TASK = makeTask("task-1", TITLE, "TODO");
let archived: boolean;
let unarchives: number;
let answer: ((ok: boolean) => void) | null;

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

beforeEach(() => {
  archived = false;
  unarchives = 0;
  answer = null;
  mockToast.mockClear();
  server.use(
    http.get("/api/proxy/api/tasks", () =>
      HttpResponse.json({ tasks: archived ? [] : [TASK] }),
    ),
    http.delete("/api/proxy/api/tasks/task-1", () => {
      archived = true;
      return HttpResponse.json({ ...TASK, is_archived: true });
    }),
    http.post("/api/proxy/api/tasks/task-1/unarchive", async () => {
      unarchives += 1;
      const ok = await new Promise<boolean>((resolve) => {
        answer = resolve;
      });
      if (!ok) return HttpResponse.json({ detail: "No." }, { status: 500 });
      archived = false;
      return HttpResponse.json(TASK);
    }),
  );
});

async function archiveIt() {
  await userEvent.click(
    await screen.findByRole("button", { name: `Actions for ${TITLE}` }),
  );
  await userEvent.click(
    await screen.findByRole("menuitem", { name: "Archive" }),
  );
  const dialog = await screen.findByRole("dialog");
  await userEvent.click(
    within(dialog).getByRole("button", { name: "Archive task" }),
  );
  await vi.waitFor(() =>
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Archived" }),
    ),
  );
  const call = mockToast.mock.calls.find(([arg]) => arg.title === "Archived");
  return call?.[0] as { action: ReactNode; duration: number };
}

function card() {
  return screen.queryByRole("button", { name: /^Plan the offsite/ });
}

describe("Undo archive", () => {
  it("offers Undo for a few seconds, and Undo puts the card back at once", async () => {
    render(<Harness />);
    const toast = await archiveIt();
    expect(toast.duration).toBe(8000);
    await vi.waitFor(() => expect(card()).toBeNull());

    render(<>{toast.action}</>);
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(
      await screen.findByRole("button", { name: /^Plan the offsite/ }),
    ).toBeDefined();
    await vi.waitFor(() => expect(unarchives).toBe(1));
    answer?.(true);
    await vi.waitFor(() => expect(archived).toBe(false));
  });

  it("takes the card away again when the task can't come back", async () => {
    render(<Harness />);
    const toast = await archiveIt();
    await vi.waitFor(() => expect(card()).toBeNull());

    render(<>{toast.action}</>);
    await userEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(
      await screen.findByRole("button", { name: /^Plan the offsite/ }),
    ).toBeDefined();
    await vi.waitFor(() => expect(answer).not.toBeNull());
    answer?.(false);

    await vi.waitFor(() => expect(card()).toBeNull());
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't bring the task back" }),
    );
  });
});
