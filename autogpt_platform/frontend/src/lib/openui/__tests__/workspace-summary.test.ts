import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "../parse";
import { connected } from "./connected-fixtures";
import { planning } from "./expanded-fixtures";

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
    expect(summary).not.toContain("amount: 0");
  });

  it.each([
    { nights: "null", field: "Unknown / not supplied" },
    { nights: "0", field: "0" },
  ])(
    "distinguishes an unknown number from an explicitly supplied zero: $nights",
    ({ nights, field }) => {
      const source = connected
        .replace(
          '"Where would you stay?", ""',
          '"Where would you stay?", "city"',
        )
        .replace('"Nights", 1, 1,', `"Nights", ${nights}, 0,`);
      const summary = workspaceText(source);
      expect(summary).toContain(`Nights: ${field}`);
    },
  );

  it("summarizes the selected labels without exposing internal ids", () => {
    const summary = workspaceText(
      connected.replace('["art"]', '["art", "history"]'),
    );
    expect(summary).toContain("Interests: Art, History");
    expect(summary).not.toContain('"history"');
  });

  it("includes Prose markdown as text", () => {
    const summary = workspaceText(
      'root = Workspace("Notes", "", [Prose("## Packing\\n\\n- Passport\\n- Charger")])',
    );
    expect(summary).toContain("## Packing\n\n- Passport\n- Charger");
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
    connected.replace('id:"coast"', 'id:"city"'),
    connected.replace('"Nights", 1, 1,', '"Nights", 0, 1,'),
  ])(
    "uses a safe fallback instead of presenting invalid output as complete",
    (source) => {
      expect(workspaceText(source)).toBe(
        "A text summary will be available when this workspace finishes rendering.",
      );
    },
  );
});
