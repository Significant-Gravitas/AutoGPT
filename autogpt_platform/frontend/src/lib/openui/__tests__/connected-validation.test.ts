import { describe, expect, it } from "vitest";
import { parseWorkspace } from "../parse";
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
      connected.replace('NumberField("nights"', 'NumberField("stay_amount"'),
      /collid|unique/i,
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
});
