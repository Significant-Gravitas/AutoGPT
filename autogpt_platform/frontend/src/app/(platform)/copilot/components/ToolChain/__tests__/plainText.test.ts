import { describe, expect, it } from "vitest";
import { looksTechnical } from "../plainText";

describe("looksTechnical", () => {
  it.each([
    "Patching brief.md with sed",
    "Editing ~/workspace/dewleaf/brief",
    "Reading workspace/dewleaf/notes",
    "Running `npm install`",
    "Calling run_capability",
    "Saving 2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11",
    "cd dewleaf && ls",
  ])("flags %j", (text) => {
    expect(looksTechnical(text)).toBe(true);
  });

  it.each([
    "Updating your launch brief",
    "Checking the content calendar dates",
    "Planning the 24/7 support rota",
    "Pricing at Rs 899 for 30ml",
    "Drafting Instagram posts",
  ])("passes %j", (text) => {
    expect(looksTechnical(text)).toBe(false);
  });
});
