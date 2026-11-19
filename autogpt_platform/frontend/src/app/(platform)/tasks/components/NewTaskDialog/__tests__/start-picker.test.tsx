import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NewTaskDialog } from "../NewTaskDialog";
import { defaultStartOn } from "../startChoice";

const NOW = new Date(2026, 10, 19, 14, 20, 0);
const MINUTE = 60_000;
let bodies: Record<string, unknown>[];

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  window.sessionStorage.clear();
  bodies = [];
  server.use(
    http.post("/api/proxy/api/tasks", async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      bodies.push(body);
      return HttpResponse.json(
        { ...makeTask("t9", "Weekly report", "TODO"), start_at: body.start_at },
        { status: 201 },
      );
    }),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

afterEach(() => {
  vi.useRealTimers();
});

async function openStart() {
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<NewTaskDialog isOpen onOpenChange={vi.fn()} experts={[]} />);
  await user.click(await screen.findByRole("button", { name: "Options" }));
  return user;
}

describe("Start picker", () => {
  it("defaults a picked day to the next round hour today, or 09:00 later", () => {
    const today = new Date(2026, 10, 19, 0, 0);
    const friday = new Date(2026, 10, 21, 0, 0);

    expect(defaultStartOn(today, NOW.getTime())).toEqual(
      new Date(2026, 10, 19, 15, 0),
    );
    expect(defaultStartOn(friday, NOW.getTime())).toEqual(
      new Date(2026, 10, 21, 9, 0),
    );
  });

  it("says a time is still to pick, and offers no past day", async () => {
    const user = await openStart();

    await user.click(
      screen.getByRole("button", { name: "Start: Starts now. Change start" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", {
        name: "Pick a date and time",
      }),
    );

    expect(
      screen.getByRole("button", {
        name: "Start: Starts: pick a time. Change start",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Starts now/ })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Start time" }));
    const yesterday = await screen.findByRole("button", {
      name: /November 18th, 2026/,
    });
    expect(yesterday).toHaveProperty("disabled", true);
    const tomorrow = screen.getByRole("button", {
      name: /November 20th, 2026/,
    });
    expect(tomorrow).toHaveProperty("disabled", false);
  });

  it("shows when a quick start lands, and resolves it at submit", async () => {
    const user = await openStart();

    await user.click(
      screen.getByRole("button", { name: "Start: Starts now. Change start" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: /^In 1 hour/ }),
    );

    expect(
      screen.getByRole("button", {
        name: /^Start: Starts in 1 hour · 3:20\sPM\. Change start$/,
      }),
    ).toBeDefined();

    await user.type(screen.getByLabelText("What needs doing?"), "Report");
    vi.setSystemTime(NOW.getTime() + 30 * MINUTE);
    await user.click(screen.getByRole("button", { name: "Create task" }));

    await vi.waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0].start_at).toBe(
      new Date(NOW.getTime() + 90 * MINUTE).toISOString(),
    );
  });
});
