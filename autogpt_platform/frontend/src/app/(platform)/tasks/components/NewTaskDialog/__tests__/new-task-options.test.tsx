import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NewTaskDialog } from "../NewTaskDialog";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

let bodies: Record<string, unknown>[];
let runs: number;
let refusal: string | null;

beforeEach(() => {
  bodies = [];
  runs = 0;
  refusal = null;
  mockToast.mockClear();
  server.use(
    http.post("/api/proxy/api/tasks", async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      bodies.push(body);
      if (refusal) {
        return HttpResponse.json({ detail: refusal }, { status: 422 });
      }
      return HttpResponse.json(
        {
          ...makeTask("task-9", "Weekly report", "TODO"),
          start_at: body.start_at ?? null,
        },
        { status: 201 },
      );
    }),
    http.post("/api/proxy/api/tasks/task-9/run", () => {
      runs += 1;
      return HttpResponse.json(
        makeTask("task-9", "Weekly report", "IN_PROGRESS"),
      );
    }),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

async function openOptions(onCreated = vi.fn()) {
  render(
    <NewTaskDialog
      isOpen
      onOpenChange={vi.fn()}
      experts={[]}
      onCreated={onCreated}
    />,
  );
  await userEvent.type(
    await screen.findByLabelText("What needs doing?"),
    "Weekly report",
  );
  await userEvent.click(screen.getByRole("button", { name: "Options" }));
  return onCreated;
}

async function pick(chip: string, item: string) {
  await userEvent.click(screen.getByRole("button", { name: chip }));
  await userEvent.click(
    await screen.findByRole("menuitemradio", { name: item }),
  );
}

function tomorrowAtNine() {
  const day = new Date();
  day.setDate(day.getDate() + 1);
  day.setHours(9, 0, 0, 0);
  return day.toISOString();
}

describe("New task options", () => {
  it("is one line until Options opens it", async () => {
    render(<NewTaskDialog isOpen onOpenChange={vi.fn()} experts={[]} />);

    const toggle = await screen.findByRole("button", { name: "Options" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("group", { name: "Task options" })).toBeNull();
  });

  it("saves a task that starts tomorrow, without starting it, and says when", async () => {
    const onCreated = await openOptions();

    await pick("Start: Starts now. Change start", "Tomorrow 9:00");
    await userEvent.type(
      screen.getByLabelText("Done when"),
      "Sent to the team",
    );
    await userEvent.click(screen.getByRole("button", { name: "Create task" }));

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalledWith("task-9"));
    expect(bodies[0]).toEqual(
      expect.objectContaining({
        title: "Weekly report",
        start_at: tomorrowAtNine(),
        done_when: "Sent to the team",
      }),
    );
    expect(bodies[0]).not.toHaveProperty("repeat");
    expect(runs).toBe(0);
    expect(mockToast).toHaveBeenCalledWith({
      title: expect.stringMatching(/^Saved for Otto: starts Tomorrow, 9:00/),
    });
  });

  it("builds the repeat from the picks, and a start time replaces it", async () => {
    const onCreated = await openOptions();

    await pick("Repeat: Doesn't repeat. Change repeat", "Every week");
    await pick("Day: Monday. Change day", "Friday");
    fireEvent.change(screen.getByLabelText("Repeat time"), {
      target: { value: "17:30" },
    });
    expect(
      screen.getByRole("button", {
        name: "Repeat: Every Friday at 17:30. Change repeat",
      }),
    ).toBeDefined();

    await pick("Start: Starts now. Change start", "In 1 hour");
    expect(
      screen.getByRole("button", {
        name: "Repeat: Doesn't repeat. Change repeat",
      }),
    ).toBeDefined();
    await pick("Repeat: Doesn't repeat. Change repeat", "Weekdays");
    expect(
      screen.getByRole("button", { name: "Start: Starts now. Change start" }),
    ).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Create task" }));

    await vi.waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(bodies[0].repeat).toEqual({ crons: ["0 9 * * 1-5"] });
    expect(bodies[0]).not.toHaveProperty("start_at");
    expect(runs).toBe(0);
    expect(mockToast).toHaveBeenCalledWith({
      title: "Saved for Otto: Every weekday at 09:00",
    });
  });

  it("shows the server's refusal under the part it is about and stays open", async () => {
    refusal = "Each cron has 5 fields: minute hour day month weekday.";
    const onCreated = await openOptions();
    await pick("Repeat: Doesn't repeat. Change repeat", "Every hour");

    await userEvent.click(screen.getByRole("button", { name: "Create task" }));

    const options = await screen.findByRole("group", { name: "Task options" });
    expect(await within(options).findByRole("alert")).toHaveProperty(
      "textContent",
      refusal,
    );
    expect(onCreated).not.toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
    expect(screen.getByLabelText("What needs doing?")).toHaveProperty(
      "value",
      "Weekly report",
    );
  });
});
