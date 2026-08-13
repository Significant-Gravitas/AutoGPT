import { describe, expect, it } from "vitest";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import {
  formatBriefingWindowStart,
  formatCurrency,
  formatDuration,
  formatHeaderDate,
  getHomeStatusLine,
  getTimeOfDayGreeting,
} from "./helpers";

describe("getTimeOfDayGreeting", () => {
  it("maps hours to greetings", () => {
    expect(getTimeOfDayGreeting(8)).toBe("Good morning");
    expect(getTimeOfDayGreeting(14)).toBe("Good afternoon");
    expect(getTimeOfDayGreeting(21)).toBe("Good evening");
  });
});

describe("getHomeStatusLine", () => {
  const dashboard = {
    attention: [{}, {}],
    active_tasks: [{}],
  } as HomeDashboardResponse;

  it("leads with decisions and active work", () => {
    expect(getHomeStatusLine(dashboard)).toBe(
      "2 decisions waiting · 1 agent is working now",
    );
  });

  it("reports an all-clear state", () => {
    expect(
      getHomeStatusLine({
        ...dashboard,
        attention: [],
        active_tasks: [],
      }),
    ).toBe("Nothing needs you right now");
  });
});

describe("formatters", () => {
  it("formats the briefing day in its configured timezone", () => {
    expect(
      formatHeaderDate(
        new Date("2026-08-09T23:30:00Z"),
        "Asia/Kolkata",
        "en-US",
      ),
    ).toEqual({ weekday: "Monday", calendarDate: "August 10" });
  });

  it("names the start of the briefing window", () => {
    expect(
      formatBriefingWindowStart(
        new Date("2026-08-09T03:30:00Z"),
        "Asia/Kolkata",
        "en-US",
      ),
    ).toBe("Sunday 9:00 AM");
  });

  it("formats measured runtime and cost", () => {
    expect(formatDuration(18_120)).toBe("5h 2m");
    expect(formatDuration(120)).toBe("2m");
    expect(formatCurrency(367)).toBe("$3.67");
  });
});
