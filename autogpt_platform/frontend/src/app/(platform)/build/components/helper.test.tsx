import { describe, expect, it } from "vitest";
import { formatAgentExecutorTitle } from "./helper";

describe("formatAgentExecutorTitle", () => {
  it("returns null when name is missing", () => {
    expect(formatAgentExecutorTitle(undefined, 3)).toBeNull();
    expect(formatAgentExecutorTitle("", 3)).toBeNull();
    expect(formatAgentExecutorTitle("   ", 3)).toBeNull();
  });

  it("returns the name when version is missing", () => {
    expect(formatAgentExecutorTitle("Content Writer", undefined)).toBe(
      "Content Writer",
    );
  });

  it("formats name with version when both are available", () => {
    expect(formatAgentExecutorTitle("Content Writer", 7)).toBe(
      "Content Writer v7",
    );
  });

  it("trims name before formatting", () => {
    expect(formatAgentExecutorTitle("  Content Writer  ", 2)).toBe(
      "Content Writer v2",
    );
  });
});
