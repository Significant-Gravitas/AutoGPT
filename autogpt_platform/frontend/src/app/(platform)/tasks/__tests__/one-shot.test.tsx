import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import {
  onlineManager,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTaskActions } from "../useTaskActions";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

let runs: number;
let archives: number;
const client: { current: QueryClient | null } = { current: null };

function pausedCount() {
  return (
    client.current
      ?.getMutationCache()
      .getAll()
      .filter((mutation) => mutation.state.isPaused).length ?? 0
  );
}

function Harness() {
  const actions = useTaskActions("task-1");
  const queryClient = useQueryClient();
  useEffect(() => {
    client.current = queryClient;
  }, [queryClient]);
  return (
    <>
      <button type="button" onClick={() => actions.run()}>
        Start
      </button>
      <button type="button" onClick={() => actions.archiveTask()}>
        Archive
      </button>
      <button type="button" onClick={() => actions.rename("Renamed")}>
        Rename
      </button>
    </>
  );
}

function clickTwice(name: string) {
  const button = screen.getByRole("button", { name });
  act(() => {
    button.click();
    button.click();
  });
}

beforeEach(() => {
  runs = 0;
  archives = 0;
  mockToast.mockClear();
  server.use(
    http.post("/api/proxy/api/tasks/task-1/run", async () => {
      runs += 1;
      await new Promise((resolve) => setTimeout(resolve, 30));
      return runs === 1
        ? HttpResponse.json(makeTask("task-1", "Plan", "IN_PROGRESS"))
        : HttpResponse.json(
            { detail: "It is already being worked on." },
            { status: 409 },
          );
    }),
    http.delete("/api/proxy/api/tasks/task-1", async () => {
      archives += 1;
      await new Promise((resolve) => setTimeout(resolve, 30));
      return HttpResponse.json(makeTask("task-1", "Plan", "TODO"));
    }),
  );
});

afterEach(() => {
  onlineManager.setOnline(true);
});

describe("One-shot task actions", () => {
  it("starts once on a double click, with no error", async () => {
    render(<Harness />);

    clickTwice("Start");

    await vi.waitFor(() => expect(runs).toBe(1));
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(runs).toBe(1);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("archives once on a double click", async () => {
    render(<Harness />);

    clickTwice("Archive");

    await vi.waitFor(() => expect(archives).toBe(1));
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(archives).toBe(1);
  });

  it("treats a 409 on a start that is already running as done", async () => {
    server.use(
      http.post("/api/proxy/api/tasks/task-1/run", () => {
        runs += 1;
        return HttpResponse.json(
          { detail: "It is already being worked on." },
          { status: 409 },
        );
      }),
    );
    render(<Harness />);

    await userEvent.click(screen.getByRole("button", { name: "Start" }));

    await vi.waitFor(() => expect(runs).toBe(1));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("still says so when the chat is busy with another message", async () => {
    server.use(
      http.post("/api/proxy/api/tasks/task-1/run", () =>
        HttpResponse.json(
          {
            detail:
              "The task's chat is busy with another message. Try again in a moment.",
          },
          { status: 409 },
        ),
      ),
    );
    render(<Harness />);

    await userEvent.click(screen.getByRole("button", { name: "Start" }));

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          description:
            "The task's chat is busy with another message. Try again in a moment.",
        }),
      ),
    );
  });

  it("fails at once offline, with one Retry toast, and leaves nothing to replay", async () => {
    server.use(
      http.patch("/api/proxy/api/tasks/task-1", () => HttpResponse.error()),
    );
    render(<Harness />);
    onlineManager.setOnline(false);

    await userEvent.click(screen.getByRole("button", { name: "Rename" }));

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't reach AutoGPT",
          variant: "destructive",
        }),
      ),
    );
    expect(mockToast).toHaveBeenCalledTimes(1);
    expect(mockToast.mock.calls[0][0].action).toBeDefined();
    expect(pausedCount()).toBe(0);
  });
});
