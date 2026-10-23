import { describe, expect, it } from "vitest";
import { describeData, summarizeData } from "../commandOutput";

describe("summarizeData", () => {
  it("reads CSV with quoted commas as one table", () => {
    const csv = 'name,note\n"Rs 899, 30ml",launch\nserum,"vitamin C, 15%"';
    expect(summarizeData(csv)).toEqual({
      kind: "table",
      rows: 2,
      columns: ["name", "note"],
    });
  });

  it("reads TSV", () => {
    expect(summarizeData("date\tpost\n2026-11-25\tLaunch")).toEqual({
      kind: "table",
      rows: 1,
      columns: ["date", "post"],
    });
  });

  it("leaves prose, a single line and ragged rows alone", () => {
    expect(
      summarizeData("Done. Updated the brief, then the calendar.\nAll good."),
    ).toBeNull();
    expect(summarizeData("a,b,c")).toBeNull();
    expect(summarizeData("a,b\n1,2,3")).toBeNull();
    expect(summarizeData("   ")).toBeNull();
    expect(
      summarizeData(
        "This sentence is far too long to be a column name, honestly\nx,y",
      ),
    ).toBeNull();
  });

  it("reads JSON objects and lists, and ignores what only looks like JSON", () => {
    expect(summarizeData('{"a": 1, "b": 2}')).toEqual({
      kind: "json",
      fields: ["a", "b"],
    });
    expect(summarizeData("[1, 2, 3]")).toEqual({ kind: "json-list", items: 3 });
    expect(summarizeData("{not json")).toBeNull();
  });
});

describe("describeData", () => {
  it("names a table by its rows and first columns", () => {
    expect(
      describeData({
        kind: "table",
        rows: 1,
        columns: ["a", "b", "c", "d", "e"],
      }),
    ).toEqual({ title: "Table", detail: "1 row · a, b, c, d, …" });
  });

  it("names JSON by its fields or items", () => {
    expect(describeData({ kind: "json", fields: ["launch"] })).toEqual({
      title: "Data",
      detail: "1 field · launch",
    });
    expect(describeData({ kind: "json", fields: [] }).detail).toBe("No fields");
    expect(describeData({ kind: "json-list", items: 2 })).toEqual({
      title: "Data",
      detail: "2 items",
    });
  });
});
