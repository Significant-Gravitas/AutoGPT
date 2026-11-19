import {
  makeExpert,
  makeTask,
} from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskMenu } from "../TaskMenu";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TASK: Task = {
  ...makeTask("task-1", "Visit clinic A", "DONE", "maria"),
  description: "Visit clinic A\nCheck the stock and the fridge log.",
  goal: "Restock before Monday",
  done_when: "The order is placed",
  start_at: new Date("2026-12-01T09:00:00Z"),
};

let bodies: Record<string, unknown>[];

beforeEach(() => {
  bodies = [];
  mockToast.mockClear();
  window.sessionStorage.clear();
  server.use(
    getListExpertsMockHandler200([makeExpert("maria", "Maria")]),
    http.post("/api/proxy/api/tasks", async ({ request }) => {
      bodies.push((await request.json()) as Record<string, unknown>);
      return HttpResponse.json(makeTask("task-2", "Visit clinic A", "TODO"), {
        status: 201,
      });
    }),
    http.post("/api/proxy/api/tasks/task-2/run", () =>
      HttpResponse.json(makeTask("task-2", "Visit clinic A", "IN_PROGRESS")),
    ),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

async function choose(item: string, prepare?: () => void) {
  // user-event puts its own clipboard in place; stub after it.
  const user = userEvent.setup();
  prepare?.();
  const onDuplicated = vi.fn();
  render(
    <TaskMenu task={TASK} onRename={vi.fn()} onDuplicated={onDuplicated} />,
  );
  await user.click(
    screen.getByRole("button", { name: "Actions for Visit clinic A" }),
  );
  await user.click(await screen.findByRole("menuitem", { name: item }));
  return { user, onDuplicated };
}

describe("Task menu: Duplicate and Copy link", () => {
  it("opens the create dialog filled in from the task, without its schedule", async () => {
    const { user, onDuplicated } = await choose("Duplicate");

    const dialog = await screen.findByRole("dialog", { name: "New task" });
    const text = within(dialog).getByLabelText(
      "What needs doing?",
    ) as HTMLTextAreaElement;
    expect(text.value).toBe(TASK.description);
    expect(
      await within(dialog).findByRole("button", {
        name: "Owner: Maria. Change owner",
      }),
    ).toBeDefined();
    expect(
      (within(dialog).getByLabelText("Done when") as HTMLInputElement).value,
    ).toBe("The order is placed");
    expect(
      (within(dialog).getByLabelText("Goal") as HTMLInputElement).value,
    ).toBe("Restock before Monday");
    expect(
      within(dialog).getByRole("button", {
        name: "Start: Starts now. Change start",
      }),
    ).toBeDefined();

    await user.clear(text);
    await user.type(text, "Visit clinic B");
    await user.click(
      within(dialog).getByRole("button", { name: "Create and start task" }),
    );

    await vi.waitFor(() => expect(onDuplicated).toHaveBeenCalledWith("task-2"));
    expect(bodies[0]).toEqual(
      expect.objectContaining({
        title: "Visit clinic B",
        owner_expert_id: "maria",
        goal: "Restock before Monday",
        done_when: "The order is placed",
      }),
    );
    expect(bodies[0]).not.toHaveProperty("start_at");
    await vi.waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Actions for Visit clinic A" }),
      ),
    );
  });

  it("copies the task's link and says so", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);

    await choose("Copy link", () =>
      vi.spyOn(navigator, "clipboard", "get").mockReturnValue({
        ...navigator.clipboard,
        writeText,
      }),
    );

    const link = `${window.location.origin}/home?show=tasks&task=task-1`;
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith(link));
    expect(mockToast).toHaveBeenCalledWith({ title: "Link copied" });
  });

  it("shows the link to copy by hand when the clipboard is unavailable", async () => {
    await choose("Copy link", () =>
      vi
        .spyOn(navigator, "clipboard", "get")
        .mockReturnValue(undefined as unknown as Clipboard),
    );

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith({
        title: "Copy this link to share the task",
        description: `${window.location.origin}/home?show=tasks&task=task-1`,
      }),
    );
  });
});
