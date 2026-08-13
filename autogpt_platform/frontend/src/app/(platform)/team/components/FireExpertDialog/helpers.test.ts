import { describe, expect, test } from "vitest";
import { getFireSummary, getPauseNames, getScheduleLine } from "./helpers";

describe("getScheduleLine", () => {
  test("reads as a reassurance when nothing pauses", () => {
    expect(getScheduleLine(0)).toBe("No scheduled runs will pause.");
  });

  test("uses the singular form for one run", () => {
    expect(getScheduleLine(1)).toBe("1 scheduled run will pause.");
  });

  test("uses the plural form for several runs", () => {
    expect(getScheduleLine(3)).toBe("3 scheduled runs will pause.");
  });
});

describe("getPauseNames", () => {
  test("returns an empty list without a preview", () => {
    expect(getPauseNames(null)).toEqual([]);
  });

  test("merges schedules and triggers into one list", () => {
    expect(
      getPauseNames({
        schedule_names: ["Content Calendar"],
        trigger_names: ["Inbox watcher"],
      }),
    ).toEqual(["Content Calendar", "Inbox watcher"]);
  });
});

describe("getFireSummary", () => {
  test("counts schedules and triggers together for the pause line", () => {
    const summary = getFireSummary({
      schedule_names: ["Content Calendar", "SEO Audit"],
      trigger_names: ["Inbox watcher"],
    });
    expect(summary.names).toEqual([
      "Content Calendar",
      "SEO Audit",
      "Inbox watcher",
    ]);
    expect(summary.scheduleLine).toBe("3 scheduled runs will pause.");
  });

  test("falls back to the reassuring line without a preview", () => {
    expect(getFireSummary(null)).toEqual({
      names: [],
      scheduleLine: "No scheduled runs will pause.",
    });
  });
});
