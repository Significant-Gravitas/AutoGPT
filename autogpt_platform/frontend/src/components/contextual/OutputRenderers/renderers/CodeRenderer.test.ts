import { describe, expect, it } from "vitest";
import { codeRenderer } from "./CodeRenderer";

describe("CodeRenderer", () => {
  it("always treats metadata-tagged code as code", () => {
    expect(
      codeRenderer.canRender("**kwargs\n- bullet", {
        type: "code",
        language: "py",
      }),
    ).toBe(true);
  });

  it("does not claim markdown-heavy text without code metadata", () => {
    expect(
      codeRenderer.canRender("# Heading\n\n- one\n- two\n\n**bold**"),
    ).toBe(false);
  });
});
