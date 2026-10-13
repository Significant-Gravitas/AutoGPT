import { describe, expect, it } from "vitest";
import { columnsSchema, calculatedRows } from "../data-values";
import { chartDataError } from "../catalog-dashboard";
import { parseWorkspace } from "../parse";
import { dataDashboard } from "./capability-fixtures";

describe("honest calculated data", () => {
  it("overrides stale supplied totals and propagates missing inputs and zero divisors", () => {
    const columns = columnsSchema.parse([
      { key: "amount", label: "Amount", type: "number" },
      { key: "count", label: "Count", type: "number" },
      {
        key: "rate",
        label: "Rate",
        type: "number",
        calculation: { operation: "divide", left: "amount", right: "count" },
      },
    ]);
    const rows = calculatedRows(columns, [
      { id: "a", cells: { amount: 10, count: 2, rate: 999 } },
      { id: "b", cells: { amount: null, count: 2 } },
      { id: "c", cells: { amount: 1, count: 0 } },
    ]);
    expect(rows.map((row) => row.cells.rate)).toEqual([5, null, null]);
    expect(
      columnsSchema.safeParse(
        columns.map((column) =>
          column.key === "rate"
            ? {
                ...column,
                calculation: { operation: "add", left: "rate", right: "count" },
              }
            : column,
        ),
      ).success,
    ).toBe(false);
  });
  it("rejects duplicate series and mixed units, and pauses signed stacked data", () => {
    const source = dataDashboard();
    expect(() =>
      parseWorkspace(
        source.replace(
          'key: "cost", label: "Cost"',
          'key: "revenue", label: "Cost"',
        ),
      ),
    ).toThrow();
    expect(() =>
      parseWorkspace(source.replace('unit: "USD"}', 'unit: "EUR"}')),
    ).toThrow();
    expect(
      chartDataError(
        [
          {
            title: "Values",
            kind: "stacked",
            x: "label",
            series: [{ key: "a", label: "A" }],
            unit: "USD",
          },
        ],
        [{ key: "a", label: "A", type: "number", unit: "USD" }],
        [{ id: "a", cells: { a: -1 } }],
      ),
    ).toContain("nonnegative");
  });
});
