import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { NewTaskDialog } from "../NewTaskDialog";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const EXPERTS = [{ id: "exp-1", name: "Maya", avatar_url: null, busy: false }];

function captureCreate() {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/tasks", async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json({ id: "task-7", number: 7, title: "x" });
    }),
  );
  return bodies;
}

function renderDialog() {
  const onOpenChange = vi.fn();
  const onCreated = vi.fn();
  render(
    <NewTaskDialog
      isOpen
      onOpenChange={onOpenChange}
      experts={EXPERTS}
      onCreated={onCreated}
    />,
  );
  return { onOpenChange, onCreated };
}

describe("NewTaskDialog", () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockToast.mockReset();
  });

  test("Enter gives the task to Otto, with the first line as its title", async () => {
    const bodies = captureCreate();
    const { onOpenChange, onCreated } = renderDialog();
    const user = userEvent.setup();

    await user.type(
      screen.getByRole("textbox", { name: "What needs doing?" }),
      "Book the offsite{Shift>}{Enter}{/Shift}Somewhere near Lisbon{Enter}",
    );

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-7"));
    expect(bodies).toEqual([
      {
        title: "Book the offsite",
        description: "Book the offsite\nSomewhere near Lisbon",
        assignee: { type: "otto", id: null },
      },
    ]);
    expect(mockToast).toHaveBeenCalledWith({
      title: "T-7 created · Otto is on it",
    });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  test("the picked expert gets the task and is remembered next time", async () => {
    const bodies = captureCreate();
    renderDialog();
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Assigned to Otto/ }));
    await user.click(await screen.findByRole("menuitem", { name: /Maya/ }));
    await user.type(
      screen.getByRole("textbox", { name: "What needs doing?" }),
      "Draft the launch post",
    );
    await user.click(screen.getByRole("button", { name: "Create task" }));

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      title: "Draft the launch post",
      assignee: { type: "expert", id: "exp-1" },
    });
    expect(mockToast).toHaveBeenCalledWith({
      title: "T-7 created · Maya is on it",
    });

    renderDialog();
    expect(
      screen.getAllByRole("button", { name: /Assigned to Maya/ }).length,
    ).toBeGreaterThan(0);
  });

  test("an empty task cannot be sent", async () => {
    const bodies = captureCreate();
    renderDialog();
    const user = userEvent.setup();

    expect(screen.getByRole("button", { name: "Create task" })).toHaveProperty(
      "disabled",
      true,
    );
    await user.type(
      screen.getByRole("textbox", { name: "What needs doing?" }),
      "   {Enter}",
    );
    expect(bodies).toHaveLength(0);
  });
});
