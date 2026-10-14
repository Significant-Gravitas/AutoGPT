import { describe, expect, it } from "vitest";
import { calculateSchedule, scheduleSchema } from "../schedule";

const plan = {
  name: "plan",
  title: "Plan",
  days: [
    {
      label: "Monday",
      windows: [
        { start: "09:00", end: "12:00" },
        { start: "13:00", end: "17:00" },
      ],
    },
  ],
  tasks: [
    { id: "a", title: "A", minutes: 60, done: false, splittable: true },
    { id: "b", title: "B", minutes: 60, done: false, splittable: false },
  ],
};

describe("schedule logic shared by itineraries", () => {
  it("does not call spare capacity feasible when a contiguous activity cannot fit", () => {
    const result = calculateSchedule(
      [
        {
          label: "Monday",
          windows: [
            { start: "09:00", end: "10:00" },
            { start: "11:00", end: "12:00" },
          ],
        },
      ],
      [
        {
          id: "visit",
          title: "Museum visit",
          minutes: 90,
          done: false,
          splittable: false,
        },
      ],
    );
    expect(result).toMatchObject({
      capacity: 120,
      required: 90,
      unscheduled: 90,
      feasible: false,
    });
  });
  it("waits for opening and refuses to run dependent tasks past a missed deadline", () => {
    const result = calculateSchedule(
      [{ label: "Today", windows: [{ start: "09:00", end: "17:00" }] }],
      [
        {
          id: "museum",
          title: "Museum",
          minutes: 120,
          done: false,
          splittable: false,
          earliest: { day: 0, time: "11:00" },
          deadline: { day: 0, time: "12:00" },
        },
        {
          id: "next",
          title: "Next stop",
          minutes: 20,
          done: false,
          splittable: false,
        },
      ],
    );
    expect(result).toMatchObject({ feasible: false, unscheduled: 140 });
    expect(result?.allocations[1].conflict).toBe("Waiting for an earlier task");
  });
  it.each([
    {
      ...plan,
      days: [
        {
          label: "Monday",
          windows: [
            { start: "09:00", end: "12:00" },
            { start: "11:00", end: "17:00" },
          ],
        },
      ],
    },
    {
      ...plan,
      days: [{ label: "Monday", windows: [{ start: "09:00", end: "08:00" }] }],
    },
    { ...plan, tasks: [plan.tasks[0], { ...plan.tasks[1], id: "a" }] },
    {
      ...plan,
      tasks: [{ ...plan.tasks[0], deadline: { day: 2, time: "15:10" } }],
    },
  ])("rejects contradictory or malformed scheduling constraints", (value) => {
    expect(scheduleSchema.safeParse(value).success).toBe(false);
  });
  it("accepts a coherent plan", () => {
    expect(scheduleSchema.safeParse(plan).success).toBe(true);
  });
});
