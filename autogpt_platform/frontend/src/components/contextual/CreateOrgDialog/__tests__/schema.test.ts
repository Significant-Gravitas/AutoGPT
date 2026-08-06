import { describe, expect, it } from "vitest";

import { createOrgSchema, slugify } from "../schema";

describe("slugify", () => {
  it("lowercases and dash-separates a name", () => {
    expect(slugify("Acme Inc.")).toBe("acme-inc");
  });

  it("strips leading and trailing separators", () => {
    expect(slugify("  ~Acme~  ")).toBe("acme");
  });

  it("never produces a trailing dash after truncating a long name", () => {
    // The separator lands on the 50th character, so a naive slice leaves a
    // trailing dash and the generated slug fails schema validation.
    const slug = slugify(`${"a".repeat(49)} tail`);

    expect(slug).toBe("a".repeat(49));
    expect(slug.endsWith("-")).toBe(false);
    expect(
      createOrgSchema.safeParse({ name: "long org", slug, description: "" })
        .success,
    ).toBe(true);
  });
});
