import { describe, expect, it } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { ActivityTrail } from "../ActivityTrail";

const TASK = makeTask("task-1", "Send the launch emails", "IN_PROGRESS");

function line(
  id: string,
  text: string,
  minute: number,
  extra: Partial<TaskActivity> = {},
): TaskActivity {
  return {
    id,
    task_id: "task-1",
    created_at: new Date(`2026-11-10T09:0${minute}:00Z`),
    actor_type: "EXPERT",
    kind: "TOOL",
    text,
    is_milestone: false,
    data: {},
    ...extra,
  };
}

function renderTrail(activity: TaskActivity[]) {
  return render(
    <ActivityTrail
      task={TASK}
      subtasks={[]}
      activity={[...activity].reverse()}
      experts={[]}
    />,
  );
}

function rowOf(text: string) {
  const row = screen.getByText(text).closest("li, button, div");
  if (!(row instanceof HTMLElement)) throw new Error(`no row for ${text}`);
  return row;
}

describe("Activity trail account and skill chips", () => {
  it("shows the account and skill a step used beside its line", () => {
    renderTrail([
      line("l-1", "Drafting the emails", 1, {
        kind: "PROGRESS",
        is_milestone: true,
      }),
      line("l-2", "Opened a skill", 2, { skill: "Cold email" }),
      line("l-3", "Used Gmail to send email", 3, {
        account: { service: "google", name: "Gmail", label: "work@acme.com" },
        skill: "Cold email",
      }),
      line("l-4", "Searched the web", 4),
    ]);

    const sent = rowOf("Used Gmail to send email");
    expect(within(sent).getByText("Gmail · work@acme.com")).toBeDefined();
    expect(within(sent).getByText("Cold email")).toBeDefined();
    expect(
      within(rowOf("Opened a skill")).getByText("Cold email"),
    ).toBeDefined();
    expect(
      within(rowOf("Searched the web")).queryByText(/Account:|Skill:/),
    ).toBeNull();
  });

  it("names the service alone for an account with no label of its own", () => {
    renderTrail([
      line("l-1", "Filing the issue", 1, {
        kind: "PROGRESS",
        is_milestone: true,
        account: { service: "github", name: "GitHub" },
      }),
    ]);

    const head = rowOf("Filing the issue");
    expect(within(head).getByTitle("Account: GitHub")).toBeDefined();
    expect(within(head).queryByTitle(/Skill:/)).toBeNull();
  });
});
