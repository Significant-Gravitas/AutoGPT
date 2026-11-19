import { areaPrompts } from "@/features/home/areaPrompts";

describe("areaPrompts", () => {
  it("picks a managed expert's area from their avatar", () => {
    const casey = {
      avatar_url: "/autogpt-characters/v1.1/expert-casey/neutral/128.webp",
      categories: [],
      role: "QA Engineer",
    };
    expect(areaPrompts(casey)).toContain("Triage flaky tests");
  });

  it("falls back to the expert's stored category", () => {
    const custom = { avatar_url: null, categories: ["Sales"], role: "Closer" };
    expect(areaPrompts(custom)).toContain("Draft a cold outreach email");
  });

  it("gives an expert with no known area the general starters", () => {
    const blank = { avatar_url: null, categories: [], role: "Helper" };
    expect(areaPrompts(blank)).toEqual([
      "Plan my week",
      "Draft a follow-up email",
      "Research a topic for me",
      "Sum up a long document",
    ]);
  });
});
