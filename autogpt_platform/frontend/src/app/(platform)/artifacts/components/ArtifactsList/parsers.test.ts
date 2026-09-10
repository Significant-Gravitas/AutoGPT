import { describe, expect, it } from "vitest";
import { parseCsv, parseIcs, parseVcard } from "./parsers";

describe("parseCsv", () => {
  it("keeps every row for a complete file (trailing newline)", () => {
    const csv = "name,age\nAlice,30\nBob,25\n";
    expect(parseCsv(csv)).toEqual({
      headers: ["name", "age"],
      rows: [
        ["Alice", "30"],
        ["Bob", "25"],
      ],
    });
  });

  it("drops the trailing partial row only when truncated mid-line", () => {
    const csv = "name,age\nAlice,30\nBo"; // no trailing newline → truncated
    expect(parseCsv(csv)).toEqual({
      headers: ["name", "age"],
      rows: [["Alice", "30"]],
    });
  });

  it("respects quoted fields containing the delimiter", () => {
    const csv = '"Doe, John",42\n"Smith, Jane",37\n';
    expect(parseCsv(csv)).toEqual({
      headers: ["Doe, John", "42"],
      rows: [["Smith, Jane", "37"]],
    });
  });

  it("unescapes doubled quotes inside quoted fields", () => {
    const csv = 'quote\n"She said ""hi"""\n';
    expect(parseCsv(csv)).toEqual({
      headers: ["quote"],
      rows: [['She said "hi"']],
    });
  });

  it("returns null for empty input", () => {
    expect(parseCsv("")).toBeNull();
  });
});

describe("parseIcs", () => {
  it("extracts event summary and times", () => {
    const ics =
      "BEGIN:VEVENT\nSUMMARY:Standup\nDTSTART:20260101T090000Z\nEND:VEVENT";
    expect(parseIcs(ics)).toMatchObject({
      summary: "Standup",
      start: "20260101T090000Z",
    });
  });
});

describe("parseVcard", () => {
  it("extracts the contact name and email", () => {
    const vcf = "BEGIN:VCARD\nFN:Jane Doe\nEMAIL:jane@example.com\nEND:VCARD";
    expect(parseVcard(vcf)).toMatchObject({
      name: "Jane Doe",
      email: "jane@example.com",
    });
  });
});
