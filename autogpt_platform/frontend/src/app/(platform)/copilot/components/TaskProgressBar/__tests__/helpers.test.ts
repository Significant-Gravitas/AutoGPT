import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  countCompleted,
  getCurrentLabel,
  getCurrentTask,
  getLatestTaskList,
  isAllComplete,
  type TodoItem,
} from "../helpers";

function todo(
  content: string,
  status: TodoItem["status"],
  activeForm?: string,
): TodoItem {
  return { content, status, activeForm };
}

function messageWithTodoWrite(todos: unknown, state?: string): UIMessage {
  return {
    id: "m1",
    role: "assistant",
    parts: [{ type: "tool-TodoWrite", state, input: { todos } }],
  } as unknown as UIMessage;
}

describe("getLatestTaskList", () => {
  it("returns null when there are no TodoWrite parts", () => {
    const messages = [
      { id: "m1", role: "assistant", parts: [{ type: "text", text: "hi" }] },
    ] as unknown as UIMessage[];
    expect(getLatestTaskList(messages)).toBeNull();
  });

  it("returns the todos from the most recent TodoWrite part", () => {
    const messages = [
      messageWithTodoWrite([todo("Old", "completed")]),
      messageWithTodoWrite([
        todo("Step 1", "completed"),
        todo("Step 2", "in_progress"),
      ]),
    ];
    const result = getLatestTaskList(messages);
    expect(result).toHaveLength(2);
    expect(result?.[0].content).toBe("Step 1");
  });

  it("skips streaming (partial) TodoWrite inputs", () => {
    const messages = [
      messageWithTodoWrite([todo("Full 1", "completed")]),
      messageWithTodoWrite([todo("Partial", "completed")], "input-streaming"),
    ];
    const result = getLatestTaskList(messages);
    expect(result?.[0].content).toBe("Full 1");
  });

  it("ignores malformed todo entries", () => {
    const messages = [
      messageWithTodoWrite([
        todo("Valid", "pending"),
        { content: "no status" },
        { status: "pending" },
        "not an object",
      ]),
    ];
    const result = getLatestTaskList(messages);
    expect(result).toHaveLength(1);
    expect(result?.[0].content).toBe("Valid");
  });

  it("returns null when todos is not an array", () => {
    expect(getLatestTaskList([messageWithTodoWrite({})])).toBeNull();
  });
});

describe("countCompleted / isAllComplete", () => {
  it("counts completed items", () => {
    const todos = [
      todo("a", "completed"),
      todo("b", "in_progress"),
      todo("c", "completed"),
    ];
    expect(countCompleted(todos)).toBe(2);
  });

  it("is complete only when every item is completed", () => {
    expect(isAllComplete([todo("a", "completed")])).toBe(true);
    expect(isAllComplete([todo("a", "completed"), todo("b", "pending")])).toBe(
      false,
    );
    expect(isAllComplete([])).toBe(false);
  });
});

describe("getCurrentTask", () => {
  const todos = [
    todo("a", "completed"),
    todo("b", "in_progress"),
    todo("c", "pending"),
  ];

  it("returns the in-progress task while streaming", () => {
    expect(getCurrentTask(todos, true)?.content).toBe("b");
  });

  it("returns the first unfinished task when not streaming", () => {
    expect(getCurrentTask(todos, false)?.content).toBe("b");
  });

  it("falls back to the last task when all are completed", () => {
    const done = [todo("a", "completed"), todo("b", "completed")];
    expect(getCurrentTask(done, false)?.content).toBe("b");
  });

  it("returns null for an empty list", () => {
    expect(getCurrentTask([], true)).toBeNull();
  });
});

describe("getCurrentLabel", () => {
  it("uses activeForm for an in-progress task with one", () => {
    expect(getCurrentLabel(todo("Do it", "in_progress", "Doing it"))).toBe(
      "Doing it",
    );
  });

  it("uses content when not in progress or no activeForm", () => {
    expect(getCurrentLabel(todo("Do it", "pending", "Doing it"))).toBe("Do it");
    expect(getCurrentLabel(todo("Do it", "in_progress"))).toBe("Do it");
  });
});
