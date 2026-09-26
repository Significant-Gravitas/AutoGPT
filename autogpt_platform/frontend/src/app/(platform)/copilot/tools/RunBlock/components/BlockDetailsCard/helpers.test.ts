import { describe, expect, it } from "vitest";
import { deriveFieldTitle } from "./helpers";

describe("deriveFieldTitle", () => {
  // Each case is a title the backend drops from the LLM-facing schema, so the
  // card has to reproduce it exactly from the property name.
  it.each([
    ["repo_url", "Repo Url"],
    ["prompt", "Prompt"],
    ["max_tokens", "Max Tokens"],
    ["createdAt", "CreatedAt"],
    ["value_input_option", "Value Input Option"],
  ])("derives %s as %s", (name, expected) => {
    expect(deriveFieldTitle(name)).toBe(expected);
  });
});
