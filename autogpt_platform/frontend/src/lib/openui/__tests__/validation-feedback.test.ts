import { describe, expect, it } from "vitest";
import { parseWorkspace } from "../parse";

function workspace(section: string) {
  return `root = Workspace("Test", "Supplied data", [${section}])`;
}

describe("OpenUI validation and repair feedback", () => {
  it.each([
    [
      'DataTable("Quotes", ["Name", "Cost"], [["Kite", "$1550", "Extra"]])',
      /DataTable.*rows\.0.*2.*3/,
    ],
    [
      'DataTable("Quotes", ["Name", "Cost"], [["Kite"]])',
      /DataTable.*rows\.0.*2.*1/,
    ],
    ['DataTable("Quotes", [], [[]])', /DataTable.*columns/],
    [
      'DataTable("Quotes", ["a", "b", "c", "d", "e", "f", "g"], [])',
      /DataTable.*columns.*6/,
    ],
    [
      'Form("trip", "Trip", [Field("days", "Days", "1", ""), Field("days", "Days again", "2", "")], "Plan", "Use these values")',
      /Form.*fields.*unique/,
    ],
    ['Mystery("Hello")', /Mystery/],
    ["missingSection", /missingSection/],
  ])(
    "rejects invalid views with an actionable location: %s",
    (section, issue) => {
      expect(() => parseWorkspace(workspace(section))).toThrow(issue);
    },
  );

  it("rejects duplicate definitions instead of silently overwriting one", () => {
    const source = `${workspace("note")}\nnote = Insight("One", "First", "neutral")\nnote = Insight("Two", "Second", "neutral")`;
    expect(() => parseWorkspace(source)).toThrow(/Duplicate.*note/);
  });

  it("does not confuse quoted text or comments with definitions", () => {
    const source = `${workspace("note")}\n# note = ignored\nnote = Insight("x = y", "note = remains text", "neutral")`;
    expect(parseWorkspace(source).typeName).toBe("Workspace");
  });

  it("rejects excessive nesting before parsing", () => {
    expect(() =>
      parseWorkspace(workspace("[".repeat(1000) + "0" + "]".repeat(1000))),
    ).toThrow(/nesting/i);
  });

  it("accepts tables with matching rows, including no results", () => {
    for (const rows of ["[]", '[["Kite", "$1550"]]']) {
      expect(
        parseWorkspace(
          workspace(`DataTable("Quotes", ["Name", "Cost"], ${rows})`),
        ).typeName,
      ).toBe("Workspace");
    }
  });
});
