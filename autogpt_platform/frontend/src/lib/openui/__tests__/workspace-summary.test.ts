import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "../parse";
import { connected } from "./connected-fixtures";
import { planning } from "./expanded-fixtures";
import { launchSchedule, rehearsalSchedule } from "./schedule-fixtures";

describe("saved workspace text", () => {
  it("uses readable choices, keeps false preferences, and marks empty values as missing", () => {
    const summary = workspaceText(connected);
    expect(summary).toContain("Interests: Art");
    expect(summary).toContain("Step-free access needed: false");
    expect(summary).toContain("Anything else?: Unknown / not supplied");
    expect(summary).not.toContain("step_free");
    expect(summary).not.toContain("Update the trip using my choices.");
    expect(workspaceText(planning)).toContain("Travel mode: Walking");
    expect(workspaceText(planning)).toContain("Visit date: 2026-10-09");
  });

  it("keeps selected-option evidence and distinguishes a missing amount from zero", () => {
    const source = connected
      .replace('"Where would you stay?", ""', '"Where would you stay?", "city"')
      .replace(
        "amount:180",
        'source:{label:"Hotel quote",url:"https://example.com/quote"}',
      );
    const summary = workspaceText(source);
    expect(summary).toContain("City hotel (selected): Near the museums");
    expect(summary).toContain("Tradeoff: More noise");
    expect(summary).toContain(
      "Source: Hotel quote (https://example.com/quote)",
    );
    expect(summary).toContain("Estimated total: Unknown / incomplete inputs");
    expect(summary).not.toContain("amount: 0");
  });

  it.each([
    {
      nights: "null",
      amount: "Unknown / incomplete inputs",
      field: "Unknown / not supplied",
    },
    { nights: "0", amount: "$66.00", field: "0" },
  ])(
    "distinguishes an unknown number from an explicitly supplied zero: $nights",
    ({ nights, amount, field }) => {
      const source = connected
        .replace(
          '"Where would you stay?", ""',
          '"Where would you stay?", "city"',
        )
        .replace('"Nights", 1, 1,', `"Nights", ${nights}, 0,`);
      const summary = workspaceText(source);
      expect(summary).toContain(`Nights: ${field}`);
      expect(summary).toContain(`Estimated total: ${amount}`);
    },
  );

  it("summarizes the selected labels and computes their count without exposing internal ids", () => {
    const source = connected
      .replace('["art"]', '["art", "history"]')
      .replace(
        '"Estimated total", "stay_amount * nights + extras_total", "currency", "USD", 2',
        '"Interests selected", "count(interests) / 2", "percent", "", 0',
      );
    const summary = workspaceText(source);
    expect(summary).toContain("Interests: Art, History");
    expect(summary).toContain("Interests selected: 100%");
  });

  it("preserves reported completion, availability gaps, and allocation times", () => {
    const summary = workspaceText(launchSchedule);
    expect(summary).toContain("Monday available: 09:00–12:00, 13:00–17:00");
    expect(summary).toContain(
      "780 min remaining; 840 min available; 60 min spare.",
    );
    expect(summary).toContain(
      "Research (240 min): Monday 09:00–Monday 12:00, Monday 13:00–Monday 14:00",
    );
    const rehearsal = workspaceText(rehearsalSchedule);
    expect(rehearsal).toContain(
      "Setup: completed; excluded from remaining work",
    );
    expect(rehearsal).toContain(
      "140 min remaining; 160 min available; 20 min spare.",
    );
    expect(rehearsal).toContain(
      "Rehearsal B (40 min): Today 14:20–Today 15:00",
    );
  });

  it("does not confuse a capacity deficit with unscheduled whole tasks", () => {
    const source = `root = Workspace("Museum visits", "Illustrative", [form])
form = Form("day", "Choose a reduction", [Schedule("visits", "Requested visits", [{label:"Afternoon",windows:[{start:"13:00",end:"16:00"}]}], [{id:"first",title:"Museum one",minutes:75,done:false,splittable:false},{id:"transfer1",title:"Transfer",minutes:20,done:false,splittable:false},{id:"second",title:"Museum two",minutes:75,done:false,splittable:false},{id:"transfer2",title:"Second transfer",minutes:20,done:false,splittable:false},{id:"third",title:"Museum three",minutes:75,done:false,splittable:false}])], "Discuss plan", "Help reduce this plan")`;
    const summary = workspaceText(source);
    expect(summary).toContain(
      "265 min remaining; 180 min available; 95 min unscheduled; plan does not fit.",
    );
    expect(summary).toContain("85 min over available time.");
    expect(summary).toContain("10 min unused.");
    expect(summary).not.toContain("85 min unscheduled");
  });

  it("bounds large valid summaries and makes truncation explicit", () => {
    const detail = "Preserve this supplied observation. ".repeat(45);
    const source = `root = Workspace("Research notes", "Supplied notes", [${Array.from(
      { length: 8 },
      (_, index) =>
        `Insight("Finding ${index + 1}", ${JSON.stringify(detail)}, "neutral")`,
    ).join(", ")}])`;
    expect(parseWorkspace(source).typeName).toBe("Workspace");
    const summary = workspaceText(source);
    expect(summary).toContain("Finding 1");
    expect(summary.length).toBeLessThanOrEqual(10_000);
    expect(summary).toMatch(
      /\[Text summary shortened\. The interactive view contains the remaining details\.\]$/,
    );
  });

  it.each([
    'root = Workspace("Truncated", "", [',
    connected.replace(
      "stay_amount * nights + extras_total",
      "missing_total + 1",
    ),
    connected.replace("quantity:2,unitPrice:25", "quantity:2,unitPrice:-25"),
  ])(
    "uses a safe fallback instead of presenting invalid output as complete",
    (source) => {
      expect(workspaceText(source)).toBe(
        "A text summary will be available when this workspace finishes rendering.",
      );
    },
  );
});
