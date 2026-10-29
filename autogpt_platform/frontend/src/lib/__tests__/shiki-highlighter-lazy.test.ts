import { describe, expect, it, vi } from "vitest";

const shikiLib = vi.hoisted(() => ({ loads: 0, failNextCreate: false }));

vi.mock("shiki", async (importOriginal) => {
  shikiLib.loads += 1;
  const shiki = await importOriginal<typeof import("shiki")>();
  return {
    ...shiki,
    createHighlighter: ((...args) => {
      if (shikiLib.failNextCreate) {
        shikiLib.failNextCreate = false;
        return Promise.reject(new Error("grammar failed to load"));
      }
      return shiki.createHighlighter(...args);
    }) as typeof shiki.createHighlighter,
  };
});

describe("shiki highlighter loading", () => {
  it("answers language questions without loading the highlighter", async () => {
    const { getSupportedLanguages, isLanguageSupported, resolveLanguage } =
      await import("../shiki-highlighter");

    expect(isLanguageSupported("py")).toBe(true);
    expect(isLanguageSupported("not-a-language")).toBe(false);
    expect(resolveLanguage("ts")).toBe("typescript");
    expect(getSupportedLanguages()).toContain("python");
    expect(shikiLib.loads).toBe(0);
  });

  it("tries again after a failed load instead of keeping the failure", async () => {
    const { getShikiHighlighter } = await import("../shiki-highlighter");

    shikiLib.failNextCreate = true;
    await expect(getShikiHighlighter()).rejects.toThrow("grammar failed");

    await expect(getShikiHighlighter()).resolves.toBeDefined();
  });

  it("loads the highlighter once, on first use", async () => {
    const { getShikiHighlighter, SHIKI_THEMES } = await import(
      "../shiki-highlighter"
    );

    const [highlighter, again] = await Promise.all([
      getShikiHighlighter(),
      getShikiHighlighter(),
    ]);

    expect(again).toBe(highlighter);
    expect(shikiLib.loads).toBe(1);
    const { tokens } = highlighter.codeToTokens("const answer = 42;", {
      lang: "typescript",
      theme: SHIKI_THEMES[0],
    });
    expect(tokens[0].map((token) => token.content).join("")).toBe(
      "const answer = 42;",
    );
    expect(new Set(tokens[0].map((token) => token.color)).size).toBeGreaterThan(
      1,
    );
  });
});
