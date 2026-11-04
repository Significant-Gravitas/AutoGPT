import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { getTaskKickoff } from "../helpers";
import { TaskKickoffCard } from "../TaskKickoffCard";

describe("TaskKickoffCard", () => {
  test("shows the task a work chat starts, not the instructions behind it", () => {
    const kickoff = getTaskKickoff({
      kind: "task_kickoff",
      task_id: "task-1",
      task_key: "T-2",
      task_title: "Create 1 developer, 1 marketer expert for me",
    });
    expect(kickoff).not.toBeNull();

    render(<TaskKickoffCard kickoff={kickoff!} />);

    expect(screen.getByText("Task T-2")).toBeDefined();
    expect(
      screen.getByText("Create 1 developer, 1 marketer expert for me"),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Open task" }).getAttribute("href"),
    ).toBe("/tasks?task=task-1");
  });

  test("other messages are not task kickoffs", () => {
    expect(getTaskKickoff({ kind: "expert_kickoff" })).toBeNull();
    expect(getTaskKickoff({ kind: "task_kickoff", task_id: "t" })).toBeNull();
    expect(getTaskKickoff(null)).toBeNull();
  });
});
