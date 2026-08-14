import { describe, expect, it } from "vitest";
import {
  asTableRows,
  buildRunLink,
  cellText,
  isOutputType,
  pickPrimaryOutput,
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

  it("pickPrimaryOutput collapses a single value and skips empty pins", () => {
    expect(pickPrimaryOutput({ empty: [], result: [[{ a: 1 }]] })).toEqual([
      { a: 1 },
    ]);
    expect(pickPrimaryOutput({ result: ["a", "b"] })).toEqual(["a", "b"]);
    expect(pickPrimaryOutput({})).toBeNull();
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

  it("buildRunLink encodes ids and returns null without a library agent", () => {
    expect(buildRunLink("lib 1", "exec/2")).toBe(
      "/library/agents/lib%201?activeTab=runs&activeItem=exec%2F2",
    );
    expect(buildRunLink(null, "exec-1")).toBeNull();
  });
});
