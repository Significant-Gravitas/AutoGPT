import { onlineManager } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import { DISARM_MS } from "../components/ScheduleSidePanel/useArmedConfirm";
import SchedulesPage from "../page";

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast }),
}));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/schedules",
  useSearchParams: () => new URLSearchParams(),
}));

const API = "/api/proxy/api";
const HOUR = 3_600_000;
const NOW = Date.now();
const ROW = { selector: "button[data-id]" };

function task(overrides: Partial<Task>): Task {
  return {
    id: "task-1",
    user_id: "user-1",
    title: "Weekly report",
    status: "TODO",
    created_by: "USER",
    created_at: new Date(NOW - 48 * HOUR),
    updated_at: new Date(NOW - HOUR),
    session_ids: ["chat-1"],
    work_session_id: "chat-1",
    ...overrides,
  };
}

const REPEATING = task({
  start_at: new Date(NOW + 2 * HOUR),
  repeat: { crons: ["0 9 * * 1"], timezone: "UTC", label: "Every Monday" },
});
const ONCE = task({
  id: "task-2",
  title: "Send the board pack",
  start_at: new Date(NOW + 5 * HOUR),
});

let pauses: number;

beforeEach(() => {
  toast.mockClear();
  pauses = 0;
  server.use(
    http.get(`${API}/tasks`, () =>
      HttpResponse.json({ tasks: [REPEATING, ONCE] }),
    ),
    http.get(`${API}/schedules`, () => HttpResponse.json([])),
    http.get(`${API}/schedules/upcoming`, () =>
      HttpResponse.json({ schedules: [] }),
    ),
    http.get(`${API}/tasks/:id/record`, () =>
      HttpResponse.json({ task: REPEATING, runs: [] }),
    ),
    http.post(`${API}/tasks/task-1/repeat/pause`, () => {
      pauses += 1;
      return HttpResponse.error();
    }),
  );
});

afterEach(() => {
  act(() => onlineManager.setOnline(true));
  vi.useRealTimers();
});

async function armStop() {
  fireEvent.click(
    await screen.findByLabelText(/^Weekly report, Every Monday/, ROW),
  );
  fireEvent.click(await screen.findByText("Stop repeating"));
  expect(screen.getByText("Stop repeating for good")).toBeDefined();
}

function expectDisarmed() {
  expect(screen.queryByText("Stop repeating for good")).toBeNull();
  expect(screen.getByText("Stop repeating")).toBeDefined();
}

describe("Schedules: the two-step confirm", () => {
  it("disarms on Escape", async () => {
    render(<SchedulesPage />);
    await armStop();

    fireEvent.keyDown(document, { key: "Escape" });

    await waitFor(expectDisarmed);
  });

  it("disarms when focus moves off its buttons", async () => {
    render(<SchedulesPage />);
    await armStop();

    act(() => screen.getByText("Open task").closest("button")?.focus());

    await waitFor(expectDisarmed);
  });

  it("disarms when another schedule is opened", async () => {
    render(<SchedulesPage />);
    await armStop();

    fireEvent.click(screen.getByLabelText(/^Send the board pack, /, ROW));
    expect(await screen.findByText("Cancel")).toBeDefined();
    fireEvent.click(screen.getByLabelText(/^Weekly report, /, ROW));

    await waitFor(expectDisarmed);
  });

  it("disarms by itself after a few seconds", async () => {
    render(<SchedulesPage />);
    vi.useFakeTimers({ shouldAdvanceTime: true });
    await armStop();

    act(() => vi.advanceTimersByTime(DISARM_MS + 100));

    await waitFor(expectDisarmed);
  });
});

describe("Schedules offline", () => {
  it("puts a pause back at once and sends nothing later", async () => {
    render(<SchedulesPage />);
    const pause = await screen.findByLabelText("Weekly report: Pause", {
      selector: "button",
    });
    act(() => onlineManager.setOnline(false));

    fireEvent.click(pause);

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't reach AutoGPT" }),
      ),
    );
    await waitFor(() =>
      expect(
        screen
          .getByLabelText(/^Weekly report: (Pause|Resume)$/, {
            selector: "button",
          })
          .getAttribute("aria-checked"),
      ).toBe("true"),
    );
    act(() => onlineManager.setOnline(true));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(pauses).toBe(1);
  });
});
