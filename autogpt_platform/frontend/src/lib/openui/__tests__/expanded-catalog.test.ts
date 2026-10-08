import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "../parse";
import { places, planning } from "./expanded-fixtures";

describe("useful OpenUI components", () => {
  it.each([places, planning])(
    "accepts the expanded component catalog",
    (source) => {
      expect(parseWorkspace(source).typeName).toBe("Workspace");
    },
  );

  it("keeps locations and typed field defaults in the text alternative", () => {
    expect(workspaceText(places)).toContain("River North");
    expect(workspaceText(places)).toContain("41.8924");
    expect(workspaceText(planning)).toContain("Travel mode: walking");
    expect(workspaceText(planning)).toContain("Budget: 100");
  });

  it.each([
    places.replace("41.8924", "1000"),
    places.replace("-87.6341", "181"),
    planning.replace("value: 3}", "value: -3}"),
    planning.replace('"2026-10-09"', '"2026-02-31"'),
    planning.replace('"walking", [{label:', '"flying", [{label:'),
    planning.replace("100, 0, 1000, 10", "1100, 0, 1000, 10"),
  ])("rejects invalid geographic and part-to-whole data", (source) => {
    expect(() => parseWorkspace(source)).toThrow();
  });
});
