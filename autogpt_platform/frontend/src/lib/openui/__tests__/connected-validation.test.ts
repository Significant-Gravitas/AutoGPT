import { describe, expect, it } from "vitest";
import { parseWorkspace } from "../parse";
import { parseFormula, evaluateFormula } from "../formula";
import { connected } from "./connected-fixtures";

describe("connected view contracts", () => {
  it("accepts the documented null bounds for an unbounded numeric field", () => {
    expect(
      parseWorkspace(
        connected.replace('"Nights", 1, 1, 6, 1', '"Nights", 1, null, null, 1'),
      ).typeName,
    ).toBe("Workspace");
  });
  it("accepts a coherent connected form", () =>
    expect(parseWorkspace(connected).typeName).toBe("Workspace"));
  it.each([
    [
      connected.replace(
        'CalculatedMetric("trip"',
        'CalculatedMetric("missing"',
      ),
      /Form.*missing/i,
    ],
    [
      connected.replace(
        "stay_amount * nights + extras_total",
        "stay_amount + missing_total",
      ),
      /missing_total/,
    ],
    [
      connected.replace('NumberField("nights"', 'NumberField("stay_amount"'),
      /collid|unique/i,
    ],
    [
      connected.replace("stay_amount * nights + extras_total", "notes"),
      /numeric/i,
    ],
    [
      connected.replace("stay_amount * nights + extras_total", "count(nights)"),
      /MultiSelect/i,
    ],
    [
      connected.replace(
        "amount:140",
        'source:{url:"javascript:alert(1)",label:"Unsafe"}',
      ),
      /source|URL/i,
    ],
    [connected.replace('id:"coast"', 'id:"city"'), /distinct/i],
  ])(
    "rejects invalid connections without publishing a broken workspace",
    (source, expected) => {
      expect(() => parseWorkspace(source)).toThrow(expected);
    },
  );
  it.each([
    ["a+b", { a: 2, b: 3 }, 5],
    ["a-b", { a: 2, b: 3 }, -1],
    ["a*b+4", { a: 2, b: 3 }, 10],
    ["(a+b)*4", { a: 2, b: 3 }, 20],
    ["a/b", { a: 2, b: 4 }, 0.5],
    ["-a+b", { a: 2, b: 3 }, 1],
    ["a+b", { a: "abc", b: 2 }, null],
    ["a+b", { a: "", b: 2 }, null],
    ["a+b", { a: false, b: 2 }, null],
    ["count(interests)", { interests: ["art", "history"] }, 2],
  ])("%s calculates only valid inputs", (formula, state, expected) => {
    expect(
      evaluateFormula(
        parseFormula(formula).root,
        (name) => (state as Record<string, unknown>)[name],
      ),
    ).toBe(expected);
  });
  it.each([
    "window.alert(1)",
    "a[0]",
    "a; 2",
    "count(a,b)",
    "a+",
    "1e999",
    "(".repeat(30) + "1" + ")".repeat(30),
  ])("rejects unsafe or malformed formula %s", (formula) =>
    expect(() => parseFormula(formula)).toThrow(),
  );
  it.each(["1 / 0", "1e308 * 1e308"])(
    "does not present a plausible result for %s",
    (formula) =>
      expect(() =>
        evaluateFormula(parseFormula(formula).root, () => 0),
      ).toThrow(),
  );
});
