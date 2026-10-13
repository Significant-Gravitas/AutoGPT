import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "../parse";
import { calculateSchedule } from "../schedule";
import {
  launchSchedule,
  rehearsalSchedule,
  revisedLaunch,
} from "./schedule-fixtures";

describe("schedule constraints and matching text", () => {
  it("splits work around lunch and derives the buffer from availability", () => {
    const text = workspaceText(launchSchedule);
    expect(text).toContain(
      "780 min remaining; 840 min available; 60 min spare",
    );
    expect(text).toContain(
      "Monday 09:00–Monday 12:00, Monday 13:00–Monday 14:00",
    );
    expect(text).toContain("Delivery (60 min): Tuesday 15:00–Tuesday 16:00");
  });
  it("preserves completed history and exposes an infeasible revision", () => {
    const text = workspaceText(revisedLaunch);
    expect(text).toContain(
      "660 min remaining; 600 min available; 60 min unscheduled",
    );
    expect(text).toContain("Research: completed; excluded from remaining work");
    expect(text).not.toContain("Research (240 min): Monday 14:00");
  });
  it("derives the same 140 remaining and 20 spare minutes for the revised rehearsal", () => {
    expect(workspaceText(rehearsalSchedule)).toContain(
      "140 min remaining; 160 min available; 20 min spare",
    );
    expect(workspaceText(rehearsalSchedule)).toContain(
      "Rehearsal B (40 min): Today 14:20–Today 15:00",
    );
  });
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
    launchSchedule.replace('start:"13:00"', 'start:"11:00"'),
    launchSchedule.replace('end:"12:00"', 'end:"08:00"'),
    launchSchedule.replace('id:"design"', 'id:"research"'),
    rehearsalSchedule.replace('day:0,time:"15:10"', 'day:2,time:"15:10"'),
  ])("rejects contradictory or malformed scheduling constraints", (source) => {
    expect(() => parseWorkspace(source)).toThrow();
  });
});
