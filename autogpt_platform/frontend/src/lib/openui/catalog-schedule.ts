import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";

const clockTime = z.string().regex(/^(?:[01]\d|2[0-3]):[0-5]\d$/);
const boundary = z.object({
  day: z.number().int().min(0).max(13),
  time: clockTime,
});

export const scheduleTaskSchema = z.object({
  id: z.string().min(1).max(80),
  title: z.string().min(1).max(200),
  minutes: z.number().int().min(1).max(20_160),
  done: z.boolean(),
  splittable: z.boolean(),
  earliest: boundary.optional(),
  deadline: boundary.optional(),
});

export const Schedule = defineComponent({
  name: "Schedule",
  description:
    "An editable sequential plan and progress checklist inside Form, with computed slots, remaining duration, capacity, buffer, unscheduled work and deadline warnings. days are chronological with unique labels and nonoverlapping HH:mm windows; omit closures and elapsed time. tasks run in array order; include transfers and breaks when they consume time. Only mark done for reported completion; done tasks stay visible but consume no future time. Use splittable=true for work that can pause across windows, false for uninterrupted visits/events. Optional earliest/deadline use zero-based day and HH:mm; put verified opening/departure constraints here. Durations, order and completion update locally; changing windows needs a chat follow-up. Use neutral titles. Do not repeat these tasks in Checklist or copy computed times/totals into static text. Capacity deficit differs from whole tasks left unscheduled when gaps cannot fit them. Never invent dates, hours or travel times. Publishes name_required_minutes, name_capacity_minutes, name_buffer_minutes, name_unscheduled_minutes and name_feasible. No external scheduling or booking.",
  props: z
    .object({
      name: z.string().min(1).max(200),
      title: z.string().min(1).max(200),
      days: z
        .array(
          z.object({
            label: z.string().min(1).max(80),
            windows: z
              .array(z.object({ start: clockTime, end: clockTime }))
              .min(1)
              .max(4),
          }),
        )
        .min(1)
        .max(14),
      tasks: z.array(scheduleTaskSchema).min(1).max(12),
    })
    .superRefine(({ days, tasks }, context) => {
      if (new Set(days.map((day) => day.label)).size !== days.length)
        context.addIssue({ code: "custom", message: "Use unique day labels." });
      if (new Set(tasks.map((task) => task.id)).size !== tasks.length)
        context.addIssue({ code: "custom", message: "Use unique task ids." });
      for (const day of days) {
        let end = "00:00";
        for (const window of day.windows) {
          if (window.start < end || window.start >= window.end)
            context.addIssue({
              code: "custom",
              message:
                "Availability windows must be ordered, nonoverlapping and start before they end.",
            });
          end = window.end;
        }
      }
      for (const task of tasks) {
        for (const value of [task.earliest, task.deadline])
          if (value && value.day >= days.length)
            context.addIssue({
              code: "custom",
              message:
                "Task time constraints must reference an existing zero-based day index.",
            });
        if (
          task.earliest &&
          task.deadline &&
          (task.earliest.day > task.deadline.day ||
            (task.earliest.day === task.deadline.day &&
              task.earliest.time >= task.deadline.time))
        )
          context.addIssue({
            code: "custom",
            message: "A task must start before its deadline.",
          });
      }
    }),
  component: null,
});
