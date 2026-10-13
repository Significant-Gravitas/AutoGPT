import { useState } from "react";
import { z } from "zod/v4";
import {
  boardTask,
  taskErrors,
  type TaskBoard,
  type BoardTask,
} from "@/lib/openui/catalog-tasks";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

export function useTaskBoard(props: z.infer<typeof TaskBoard.props>) {
  const field = useFieldView(`${props.name}__draft`, props.tasks);
  const parsed = z
    .array(
      boardTask.extend({ title: z.string().max(200), due: z.string().max(10) }),
    )
    .max(30)
    .safeParse(field.value);
  const tasks = parsed.success ? parsed.data : props.tasks;
  const [actionError, setError] = useState("");
  const error =
    taskErrors(tasks)[0] ||
    (tasks.some((task) => !boardTask.safeParse(task).success)
      ? "Correct the task title or due date."
      : "");
  useDerivedField(props.name, tasks);
  useDerivedField(`${props.name}__valid`, !error);
  useDerivedField(`${props.name}_count`, tasks.length);
  useDerivedField(
    `${props.name}_done`,
    tasks.filter((task) => task.status === "done").length,
  );
  function change(task: BoardTask) {
    const next = tasks.map((item) => (item.id === task.id ? task : item));
    const structural =
      task.status !== tasks.find((item) => item.id === task.id)?.status ||
      JSON.stringify(task.dependsOn) !==
        JSON.stringify(tasks.find((item) => item.id === task.id)?.dependsOn);
    const issue = structural ? (taskErrors(next)[0] ?? "") : "";
    setError(issue);
    if (!issue) field.setValue(next);
  }
  function remove(id: string) {
    if (tasks.some((task) => task.dependsOn.includes(id))) {
      setError("Remove this task from its dependants' prerequisites first.");
      return;
    }
    field.setValue(tasks.filter((task) => task.id !== id));
    setError("");
  }
  function add() {
    if (tasks.length < 30)
      field.setValue([
        ...tasks,
        {
          id: `task-${crypto.randomUUID()}`,
          title: "New task",
          owner: "",
          due: "",
          status: "todo",
          dependsOn: [],
        },
      ]);
  }
  return { tasks, error, actionError, change, remove, add };
}
