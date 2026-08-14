import { describe, expect, it } from "vitest";
import {
  asTableRows,
  buildRunLink,
  cellText,
  isOutputType,
  pickOutputForType,
  tableColumns,
  toCsv,
} from "../helpers";

describe("WorkOutputSheet helpers", () => {
  it("isOutputType only accepts known variants", () => {
    expect(isOutputType("table")).toBe(true);
    expect(isOutputType("doc")).toBe(true);
    expect(isOutputType("image")).toBe(true);
    expect(isOutputType("unknown")).toBe(true);
    expect(isOutputType("chart")).toBe(false);
    expect(isOutputType(null)).toBe(false);
  });

  it("asTableRows accepts a non-empty list of records only", () => {
    expect(asTableRows([{ a: 1 }, { b: 2 }])).toHaveLength(2);
    expect(asTableRows([])).toBeNull();
    expect(asTableRows([1, 2])).toBeNull();
    expect(asTableRows("nope")).toBeNull();
  });

  it("pickOutputForType prefers the classified pin key", () => {
    const outputs = {
      status: ["ok"],
      results: [[{ metric: "signups" }]],
    };
    expect(pickOutputForType(outputs, "table", "results")).toEqual([
      { metric: "signups" },
    ]);
  });

  it("pickOutputForType without a key skips pins that cannot render as the type", () => {
    const outputs = {
      status: ["ok"],
      results: [[{ metric: "signups" }]],
    };
    expect(pickOutputForType(outputs, "table", null)).toEqual([
      { metric: "signups" },
    ]);
  });

  it("pickOutputForType collapses single values and returns null when nothing renders", () => {
    expect(pickOutputForType({ doc: ["# Title"] }, "doc", "doc")).toBe(
      "# Title",
    );
    expect(pickOutputForType({ n: [42] }, "table", null)).toBeNull();
    expect(pickOutputForType({}, "doc", null)).toBeNull();
  });

  it("pickOutputForType renders multi-value string pins consistently", () => {
    expect(
      pickOutputForType({ report: ["First", "Second"] }, "doc", "report"),
    ).toBe("First\n\nSecond");
    expect(
      pickOutputForType(
        { images: ["https://example.com/1.png", "https://example.com/2.png"] },
        "image",
        "images",
      ),
    ).toBe("https://example.com/1.png");
  });

  it("tableColumns unions keys across rows in first-seen order", () => {
    expect(tableColumns([{ a: 1 }, { b: 2, a: 3 }])).toEqual(["a", "b"]);
  });

  it("cellText serializes objects and blanks nullish values", () => {
    expect(cellText(null)).toBe("");
    expect(cellText(42)).toBe("42");
    expect(cellText({ a: 1 })).toBe('{"a":1}');
  });

  it("toCsv escapes commas, quotes and newlines", () => {
    const csv = toCsv([{ name: "a,b", note: 'say "hi"' }]);
    expect(csv.split("\n")[0]).toBe("name,note");
    expect(csv).toContain('"a,b"');
    expect(csv).toContain('"say ""hi"""');
  });

  it("toCsv quotes carriage returns so they cannot create a new record", () => {
    const csv = toCsv([{ value: "safe\r=FORMULA()" }]);
    expect(csv).toBe('value\n"safe\r=FORMULA()"');
  });

  it("toCsv neutralizes formula-leading string cells", () => {
    const csv = toCsv([
      {
        a: '=WEBSERVICE("https://attacker.test")',
        b: "+SUM(A1)",
        c: "-cmd",
        d: "@import",
      },
    ]);
    const cells = csv.split("\n")[1].split(",");
    // The apostrophe lands before RFC-4180 quoting wraps the cell.
    expect(cells[0]).toBe('"\'=WEBSERVICE(""https://attacker.test"")"');
    expect(cells[1]).toBe("'+SUM(A1)");
    expect(cells[2]).toBe("'-cmd");
    expect(cells[3]).toBe("'@import");
  });

  it("toCsv neutralizes formulas hidden behind leading whitespace or control chars", () => {
    const csv = toCsv([{ a: "  =1+1", b: "\t@cmd", c: "\u0000=x" }]);
    const cells = csv.split("\n")[1];
    expect(cells).toContain("'  =1+1");
    expect(cells).toContain("'\t@cmd");
    expect(cells).toContain("'\u0000=x");
  });

  it("toCsv neutralizes formula-leading headers", () => {
    const csv = toCsv([{ "=cmd|calc": "x" }]);
    expect(csv.split("\n")[0]).toBe("'=cmd|calc");
  });

  it("toCsv keeps genuine numbers numeric, including negatives", () => {
    const csv = toCsv([{ delta: -12.5, count: 3 }]);
    expect(csv.split("\n")[1]).toBe("-12.5,3");
  });

  it("toCsv respects an explicit column subset", () => {
    const csv = toCsv([{ a: 1, b: 2, c: 3 }], ["a", "b"]);
    expect(csv).toBe("a,b\n1,2");
  });

  it("buildRunLink encodes ids and returns null without a library agent", () => {
    expect(buildRunLink("lib 1", "exec/2")).toBe(
      "/library/agents/lib%201?activeTab=runs&activeItem=exec%2F2",
    );
    expect(buildRunLink(null, "exec-1")).toBeNull();
  });
});
