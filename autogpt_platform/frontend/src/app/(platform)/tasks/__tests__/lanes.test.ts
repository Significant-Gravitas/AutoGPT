import { describe, expect, test } from "vitest";
import type { Task } from "../data/types";
import { buildBoardColumns, isCancellable } from "../helpers";

function task(id: string, state: Task["state"]) {
  return { id, state, updated_at: "2026-11-04T10:00:00Z" } as Task;
}

describe("board lanes", () => {
  test("the board has a fourth lane for cancelled tasks", () => {
    const columns = buildBoardColumns([
      task("a", "needs_you"),
      task("b", "working"),
      task("c", "done"),
      task("d", "stopped"),
    ]);

    expect(columns.map((c) => [c.label, c.tasks.map((t) => t.id)])).toEqual([
      ["Needs you", ["a"]],
      ["Working", ["b"]],
      ["Done", ["c"]],
      ["Cancelled", ["d"]],
    ]);
  });

  test("only working or waiting tasks can be cancelled", () => {
    expect(
      (["working", "needs_you", "stuck", "done", "stopped"] as const).map(
        isCancellable,
      ),
    ).toEqual([true, true, true, false, false]);
  });
});
