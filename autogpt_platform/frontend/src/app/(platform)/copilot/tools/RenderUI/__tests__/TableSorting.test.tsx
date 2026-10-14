import { fireEvent, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { showResult } from "./renderConnected";

describe("numeric tables in chat", () => {
  it.each([
    [
      ["-2", "-10", "5"],
      ["-10", "-2", "5"],
    ],
    [
      ["1.2", "1.10", "1.05"],
      ["1.05", "1.10", "1.2"],
    ],
  ])(
    "sorts amounts by value without changing their display",
    async (values, expected) => {
      const rows = values.map((value) => [value]);
      await showResult(
        undefined,
        false,
        `root = Workspace("Transactions", "Supplied amounts", [data])\ndata = DataTable("Amounts", ["Amount"], ${JSON.stringify(rows)})`,
      );
      fireEvent.click(await screen.findByRole("button", { name: /Amount/ }));
      const cells = within(screen.getByRole("table")).getAllByRole("cell");
      expect(cells.map((cell) => cell.textContent)).toEqual(expected);
    },
  );
});
