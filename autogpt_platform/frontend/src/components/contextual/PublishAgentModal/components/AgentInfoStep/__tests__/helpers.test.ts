import { describe, expect, it } from "vitest";
import { publishAgentSchemaFactory } from "../helpers";

const validBase = {
  title: "My Agent",
  subheader: "A short tagline",
  slug: "my-agent",
  youtubeLink: "https://www.youtube.com/watch?v=abcdefghijk",
  category: "automation",
  description: "Does X for Y users.",
  recommendedScheduleCron: "",
  instructions: "",
  agentOutputDemo: "https://www.youtube.com/watch?v=abcdefghijk",
  changesSummary: "",
};

describe("publishAgentSchemaFactory", () => {
  it("validates a happy-path submission for first-time publish", () => {
    const result = publishAgentSchemaFactory(false).safeParse(validBase);
    expect(result.success).toBe(true);
  });

  it("rejects an empty title for first-time publish", () => {
    const result = publishAgentSchemaFactory(false).safeParse({
      ...validBase,
      title: "",
    });
    expect(result.success).toBe(false);
  });

  it("rejects slugs with disallowed characters", () => {
    const result = publishAgentSchemaFactory(false).safeParse({
      ...validBase,
      slug: "Has Spaces",
    });
    expect(result.success).toBe(false);
  });

  it("rejects a description over 1000 chars", () => {
    const result = publishAgentSchemaFactory(false).safeParse({
      ...validBase,
      description: "a".repeat(1001),
    });
    expect(result.success).toBe(false);
  });

  it("rejects instructions over 2000 chars", () => {
    const result = publishAgentSchemaFactory(false).safeParse({
      ...validBase,
      instructions: "a".repeat(2001),
    });
    expect(result.success).toBe(false);
  });

  it("rejects an invalid YouTube link", () => {
    const result = publishAgentSchemaFactory(false).safeParse({
      ...validBase,
      youtubeLink: "https://example.com/nope",
    });
    expect(result.success).toBe(false);
  });

  it("requires changesSummary on marketplace updates", () => {
    const result = publishAgentSchemaFactory(true).safeParse({
      ...validBase,
      changesSummary: "",
    });
    expect(result.success).toBe(false);
  });

  it("accepts an update with the changes summary filled in", () => {
    const result = publishAgentSchemaFactory(true).safeParse({
      ...validBase,
      changesSummary: "Fixed bug in agent",
    });
    expect(result.success).toBe(true);
  });

  it("treats title / subheader / category as optional on updates", () => {
    const result = publishAgentSchemaFactory(true).safeParse({
      ...validBase,
      title: "",
      subheader: "",
      category: "",
      changesSummary: "Refreshed copy",
    });
    expect(result.success).toBe(true);
  });
});
