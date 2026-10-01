import { describe, expect, it } from "vitest";

import { beautifyString } from "@/lib/utils";

describe("beautifyString", () => {
  it("keeps brand names that the camel-case split would break apart", () => {
    expect(beautifyString("AnySearchExtractBlock")).toBe(
      "AnySearch Extract Block",
    );
    expect(beautifyString("AllQuietGetOnCallBlock")).toBe(
      "AllQuiet Get On Call Block",
    );
  });

  it("splits ordinary camel case into words", () => {
    expect(beautifyString("CompanySearchBlock")).toBe("Company Search Block");
  });
});
