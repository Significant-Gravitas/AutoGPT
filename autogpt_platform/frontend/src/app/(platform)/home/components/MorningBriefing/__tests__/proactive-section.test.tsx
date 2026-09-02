import { getListTasksMockHandler } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { expect, test } from "vitest";
import { ProactiveSection } from "../components/ProactiveSection/ProactiveSection";

function makeTask(overrides: Partial<DelegatedTask> = {}): DelegatedTask {
  return {
    id: "task-dream-1",
    title: "Refresh the competitor pricing sheet",
    spec: "Compare pricing pages and update the sheet.",
    status: "QUEUED",
    acceptance: "PENDING",
    created_by_type: "DREAM",
    created_by_id: null,
    owner: null,
    parent_task_id: null,
    root_task_id: "task-dream-1",
    origin_session_id: null,
    ancestor_expert_ids: [],
    handoff_count: 0,
    revision_count: 0,
    spend_total: 0,
    outcome_summary: null,
    amendments: [],
    created_at: new Date("2026-08-30T09:00:00Z"),
    updated_at: new Date("2026-08-30T09:00:00Z"),
    runs: [],
    ...overrides,
  };
}

test("renders dream proposals and outcomes, hiding other origins", async () => {
  server.use(
    getListTasksMockHandler([
      makeTask(),
      makeTask({
        id: "task-dream-2",
        title: "Weekly traffic digest",
        status: "DONE",
        outcome_summary: "Digest posted to your inbox.",
      }),
      makeTask({
        id: "task-user",
        title: "A user-created task",
        created_by_type: "USER",
      }),
    ]),
  );

  render(<ProactiveSection />);

  expect(await screen.findByText("Proactive")).toBeTruthy();

  const proposal = screen.getByRole("link", {
    name: /Refresh the competitor pricing sheet/,
  });
  expect(proposal.getAttribute("href")).toBe("/team/tasks/task-dream-1");
  expect(screen.getByText(/Suggested by your team/)).toBeTruthy();

  const outcome = screen.getByRole("link", { name: /Weekly traffic digest/ });
  expect(outcome.getAttribute("href")).toBe("/team/tasks/task-dream-2");
  expect(screen.getByText("Digest posted to your inbox.")).toBeTruthy();

  expect(screen.queryByText("A user-created task")).toBeNull();
});

test("renders nothing when no dream tasks exist", async () => {
  let requested = false;
  server.use(
    getListTasksMockHandler(() => {
      requested = true;
      return [makeTask({ created_by_type: "USER" })];
    }),
  );

  const { container } = render(<ProactiveSection />);

  await waitFor(() => expect(requested).toBe(true));
  expect(container.textContent).toBe("");
});
