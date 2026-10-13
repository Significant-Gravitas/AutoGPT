import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { identifier, shortText } from "./data-values";

export const boardTask = z.object({
  id: identifier,
  title: shortText,
  owner: z.string().max(120),
  due: z.union([z.iso.date(), z.literal("")]),
  status: z.enum(["todo", "doing", "done"]),
  dependsOn: z.array(identifier).max(30),
});
export type BoardTask = z.infer<typeof boardTask>;

export function taskErrors(tasks: BoardTask[]) {
  const errors: string[] = [];
  const byID = new Map(tasks.map((task) => [task.id, task]));
  if (byID.size !== tasks.length) errors.push("Task ids must be unique.");
  const visited = new Set<string>();
  const visiting = new Set<string>();
  function cycle(id: string): boolean {
    if (visiting.has(id)) return true;
    if (visited.has(id)) return false;
    visiting.add(id);
    const found = byID.get(id)?.dependsOn.some(cycle) ?? false;
    visiting.delete(id);
    visited.add(id);
    return found;
  }
  if (tasks.some((task) => cycle(task.id)))
    errors.push("Dependencies contain a cycle.");
  for (const task of tasks) {
    if (!task.title.trim()) errors.push("Give each task a title.");
    if (
      task.dependsOn.some((id) => !byID.has(id)) ||
      new Set(task.dependsOn).size !== task.dependsOn.length
    )
      errors.push(`${task.title}: choose distinct existing dependencies.`);
    if (
      task.status !== "todo" &&
      task.dependsOn.some((id) => byID.get(id)?.status !== "done")
    )
      errors.push(`${task.title}: prerequisite work is not complete.`);
  }
  return errors;
}

export const TaskBoard = defineComponent({
  name: "TaskBoard",
  description:
    "Local task board inside Form: up to30 tasks with stable ids, title, owner, due ISO date or blank, todo/doing/done status and dependsOn ids. Users edit tasks, owners, dates and dependencies and move statuses with keyboard-accessible controls. Cycles and starting work before prerequisites are complete are blocked. This records user-reported progress; never use it as proof that agents ran. Use Schedule for sequential time capacity and ExecutionView for real backend status.",
  props: z
    .object({
      name: identifier,
      title: shortText,
      tasks: z.array(boardTask).max(30),
      allowAdd: z.boolean(),
    })
    .superRefine(({ tasks }, context) => {
      for (const message of taskErrors(tasks))
        context.addIssue({ code: "custom", message });
    }),
  component: null,
});
