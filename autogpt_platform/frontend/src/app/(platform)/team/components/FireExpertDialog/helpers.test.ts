import { describe, expect, test } from "vitest";
import { getAutomationLine, getFireSummary, getPauseItems } from "./helpers";

describe("getAutomationLine", () => {
  test("reads as a reassurance when nothing pauses", () => {
    expect(getAutomationLine(0)).toBe("No automations will pause.");
  });

  test("uses the singular form for one automation", () => {
    expect(getAutomationLine(1)).toBe("1 automation will pause.");
  });

  test("uses the plural form for several automations", () => {
    expect(getAutomationLine(3)).toBe("3 automations will pause.");
  });
});

describe("getPauseItems", () => {
  test("returns an empty list without a preview", () => {
    expect(getPauseItems(null)).toEqual([]);
  });

  test("merges schedules and triggers with type-unique keys", () => {
    expect(
      getPauseItems({
        schedule_names: ["Content Calendar"],
        trigger_names: ["Inbox watcher"],
      }),
    ).toEqual([
      { id: "schedule-0", name: "Content Calendar" },
      { id: "trigger-0", name: "Inbox watcher" },
    ]);
  });

  test("keeps keys unique when a schedule and trigger share a name", () => {
    const items = getPauseItems({
      schedule_names: ["Daily digest"],
      trigger_names: ["Daily digest"],
    });
    const keys = items.map((item) => item.id);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("getFireSummary", () => {
  test("counts schedules and triggers together as automations", () => {
    const summary = getFireSummary({
      schedule_names: ["Content Calendar", "SEO Audit"],
      trigger_names: ["Inbox watcher"],
    });
    expect(summary.items.map((item) => item.name)).toEqual([
      "Content Calendar",
      "SEO Audit",
      "Inbox watcher",
    ]);
    expect(summary.automationLine).toBe("3 automations will pause.");
  });

  test("falls back to the reassuring line without a preview", () => {
    expect(getFireSummary(null)).toEqual({
      items: [],
      automationLine: "No automations will pause.",
    });
  });
});
