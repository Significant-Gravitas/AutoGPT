import { screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { makeTask, renderTasksPage, serveTasks } from "./fixtures";

describe("Tasks list refresh", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("re-reads the list every 5 seconds until every task has settled", async () => {
    const api = serveTasks([
      makeTask({ id: "t1", title: "Research trends", status: "IN_PROGRESS" }),
    ]);
    renderTasksPage();
    await screen.findByText("Research trends");
    expect(api.listCalls).toBe(1);

    await vi.advanceTimersByTimeAsync(5_000);
    await waitFor(() => expect(api.listCalls).toBe(2));

    api.tasks = api.tasks.map((t) => ({ ...t, status: "DONE" as const }));
    await vi.advanceTimersByTimeAsync(5_000);
    await waitFor(() => expect(api.listCalls).toBe(3));
    await waitFor(() =>
      expect(screen.getByTestId("task-status").dataset.status).toBe("DONE"),
    );

    await vi.advanceTimersByTimeAsync(20_000);
    expect(api.listCalls).toBe(3);
  });

  test("doesn't poll when every task has already settled", async () => {
    const api = serveTasks([
      makeTask({ id: "t1", title: "Send the newsletter", status: "DONE" }),
      makeTask({ id: "t2", title: "Pull numbers", status: "FAILED" }),
    ]);
    renderTasksPage();
    await screen.findByText("Send the newsletter");

    await vi.advanceTimersByTimeAsync(20_000);
    expect(api.listCalls).toBe(1);
  });
});
