import type { CodeHighlighterPlugin } from "streamdown";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createSingletonCodePlugin } from "../streamdown-code-plugin";
import {
  type BundledLanguage,
  getShikiHighlighter,
  SHIKI_THEMES,
} from "../shiki-highlighter";

type HighlightResult = NonNullable<
  ReturnType<CodeHighlighterPlugin["highlight"]>
>;

const plugin = createSingletonCodePlugin();

function highlight(code: string, language: string) {
  return new Promise<HighlightResult>((resolve) => {
    const result = plugin.highlight(
      { code, language: language as BundledLanguage, themes: SHIKI_THEMES },
      resolve,
    );
    if (result) resolve(result);
  });
}

async function highlightFromScratch(code: string, lang: BundledLanguage) {
  const highlighter = await getShikiHighlighter();
  const result = highlighter.codeToTokens(code, {
    lang,
    themes: { light: SHIKI_THEMES[0], dark: SHIKI_THEMES[1] },
  });
  return {
    tokens: result.tokens.map((line) =>
      line.map((token) => ({
        content: token.content,
        color: token.color,
        htmlStyle: token.htmlStyle,
      })),
    ),
    fg: result.fg,
    bg: result.bg,
  };
}

function prefixes(code: string, count: number) {
  const step = Math.max(1, Math.ceil(code.length / count));
  const result: string[] = [];
  for (let end = step; end < code.length; end += step) {
    result.push(code.slice(0, end));
  }
  result.push(code);
  return result;
}

async function spyOnTokenizer() {
  const highlighter = await getShikiHighlighter();
  return vi.spyOn(highlighter, "codeToTokens");
}

function tokenizedChars(spy: Awaited<ReturnType<typeof spyOnTokenizer>>) {
  return spy.mock.calls.reduce((sum, [code]) => sum + code.length, 0);
}

const SAMPLES: { lang: BundledLanguage; code: string }[] = [
  {
    lang: "typescript",
    code: [
      "/**",
      " * A block comment that spans",
      " * several lines.",
      " */",
      "export function greet(name: string) {",
      "  const message = `Hello, ${name}!",
      "  Welcome back.`;",
      '  return message + " " + "done";',
      "}",
      "",
      "",
      "const value = 42; // trailing comment",
    ].join("\n"),
  },
  {
    lang: "python",
    code: [
      "def greet(name):",
      '    """',
      "    A docstring that spans",
      "    several lines.",
      '    """',
      "    return f'Hello, {name}!'",
      "",
      "class Point:",
      "    x: int = 0",
    ].join("\n"),
  },
  {
    lang: "html",
    code: [
      "<html>",
      "  <style>",
      "    body { color: red; }",
      "  </style>",
      "  <script>",
      "    const answer = 42;",
      "  </script>",
      "  <!-- a comment",
      "  that keeps going -->",
      "</html>",
    ].join("\n"),
  },
  {
    lang: "markdown",
    code: [
      "# Title",
      "",
      "```python",
      "print('nested')",
      "```",
      "",
      "Some *text* after.",
    ].join("\n"),
  },
  {
    lang: "json",
    code: '{\r\n  "a": [1, 2, 3],\r\n  "b": { "c": "d" }\r\n}\r\n',
  },
];

// Streamdown hands the plugin an open fence's body with a newline after the
// partial last line, so a still-growing line already ends in "\n".
const SHAPES = [
  { shape: "raw prefixes", wrap: (text: string) => text },
  { shape: "Streamdown fence bodies", wrap: (text: string) => `${text}\n` },
];

const CASES = SAMPLES.flatMap((sample) =>
  SHAPES.map((shape) => ({ ...sample, ...shape })),
);

afterEach(() => {
  vi.restoreAllMocks();
});

describe("streamdown code plugin while a block streams", () => {
  it.each(CASES)(
    "gives every streamed version of a $lang block ($shape) the same tokens as highlighting it from scratch",
    async ({ lang, code, wrap }) => {
      for (let end = 1; end <= code.length; end++) {
        const partial = wrap(code.slice(0, end));
        const streamed = await highlight(partial, lang);
        expect(streamed).toEqual(await highlightFromScratch(partial, lang));
      }
    },
    60_000,
  );

  it("gives each of many in-flight versions its own tokens", async () => {
    const code = SAMPLES[0].code.replace("greet", "greetInFlight");
    const versions = prefixes(code, 30);
    const results = await Promise.all(
      versions.map((version) => highlight(version, "typescript")),
    );
    for (const [index, version] of versions.entries()) {
      expect(results[index]).toEqual(
        await highlightFromScratch(version, "typescript"),
      );
    }
  });

  it.each(SHAPES)(
    "tokenizes each line of a growing block about once, not once per flush ($shape)",
    async ({ shape, wrap }) => {
      const lines = Array.from(
        { length: 80 },
        (_, i) => `const value${i} = compute(${i}, "${shape} line ${i}");`,
      );
      const code = lines.join("\n");
      const spy = await spyOnTokenizer();

      for (const version of prefixes(code, 50)) {
        await highlight(wrap(version), "ts");
      }

      // Re-tokenizing every version from scratch would cost ~25x the block.
      expect(tokenizedChars(spy)).toBeLessThan(code.length * 2);
      expect(await highlight(wrap(code), "ts")).toEqual(
        await highlightFromScratch(wrap(code), "typescript"),
      );
    },
  );

  it.each(SHAPES)(
    "keeps a finished block's tokens cached through 600 updates of another block ($shape)",
    async ({ shape, wrap }) => {
      const closed = wrap(`print("finished before the next one: ${shape}")`);
      const closedResult = await highlight(closed, "python");

      // 60 lines of 10 words, one word per update, as the copilot streams.
      let done = "";
      for (let line = 0; line < 60; line++) {
        const words = Array.from({ length: 10 }, (_, i) => `w${line}_${i}`);
        for (let count = 1; count <= words.length; count++) {
          const partial = words.slice(0, count).join(" ");
          await highlight(wrap(`${done}${partial}`), "javascript");
        }
        done = `${done}${words.join(" ")}\n`;
      }

      const spy = await spyOnTokenizer();
      const again = plugin.highlight(
        {
          code: closed,
          language: "python" as BundledLanguage,
          themes: SHIKI_THEMES,
        },
        () => undefined,
      );
      expect(again).toBe(closedResult);
      expect(spy).not.toHaveBeenCalled();
    },
    60_000,
  );

  it("still returns the cached tokens of an earlier version requested again", async () => {
    const first = "const shared = 1;\n";
    const second = `${first}const extended = 2;`;
    const firstResult = await highlight(first, "typescript");
    await highlight(second, "typescript");

    const again = plugin.highlight(
      {
        code: first,
        language: "typescript" as BundledLanguage,
        themes: SHIKI_THEMES,
      },
      () => undefined,
    );
    expect(again).toBe(firstResult);
  });

  it("falls back to a full highlight when resuming from a checkpoint fails", async () => {
    const code = "const resumed = 1;\nconst after = 2;\n";
    await highlight(code, "typescript");
    const highlighter = await getShikiHighlighter();
    const original = highlighter.codeToTokens.bind(highlighter);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.spyOn(highlighter, "codeToTokens").mockImplementation(
      (input, options) => {
        if ("grammarState" in options && options.grammarState) {
          throw new Error("bad grammar state");
        }
        return original(input, options);
      },
    );

    const longer = `${code}const more = 3;`;
    expect(await highlight(longer, "typescript")).toEqual(
      await highlightFromScratch(longer, "typescript"),
    );
    expect(warn).toHaveBeenCalled();
  });
});
