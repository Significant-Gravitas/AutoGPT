import { render, waitFor } from "@testing-library/react";
import { Streamdown } from "streamdown";
import { describe, expect, it } from "vitest";

import { code } from "../streamdown-code-plugin";
import { getShikiHighlighter, SHIKI_THEMES } from "../shiki-highlighter";

const SOURCE = [
  "/**",
  " * Rendered while it streams.",
  " */",
  "export function total(rows: { amount: number }[]) {",
  "  const label = `rows: ${rows.length}`;",
  "  return rows.reduce((sum, row) => sum + row.amount, 0);",
  "}",
].join("\n");

function markdown(body: string) {
  return `Here is the script.\n\n\`\`\`ts\n${body}`;
}

function renderedLines(container: HTMLElement) {
  const body = container.querySelector('[data-streamdown="code-block-body"]');
  return Array.from(body?.querySelectorAll("code > span") ?? []).map((line) =>
    Array.from(line.children).map((token) => ({
      content: token.textContent,
      color: (token as HTMLElement).style.color,
    })),
  );
}

async function expectedLines(source: string) {
  const highlighter = await getShikiHighlighter();
  return highlighter
    .codeToTokens(source, {
      lang: "typescript",
      themes: { light: SHIKI_THEMES[0], dark: SHIKI_THEMES[1] },
    })
    .tokens.map((line) =>
      line.map((token) => ({
        content: token.content,
        color: token.htmlStyle?.color ?? token.color,
      })),
    );
}

describe("Streamdown with the shiki code plugin", () => {
  it("shows the fully highlighted block once a streamed code block is complete", async () => {
    const { container, rerender } = render(
      <Streamdown plugins={{ code }} isAnimating>
        {markdown("")}
      </Streamdown>,
    );

    for (let end = 1; end <= SOURCE.length; end += 7) {
      rerender(
        <Streamdown plugins={{ code }} isAnimating>
          {markdown(SOURCE.slice(0, end))}
        </Streamdown>,
      );
    }
    rerender(
      <Streamdown plugins={{ code }} isAnimating={false}>
        {`${markdown(SOURCE)}\n\`\`\`\n\nDone.`}
      </Streamdown>,
    );

    // Streamdown hands the plugin the fence body with its closing newline.
    const expected = await expectedLines(`${SOURCE}\n`);
    await waitFor(() => expect(renderedLines(container)).toEqual(expected));
    expect(expected.flat().every((token) => token.color)).toBe(true);
  });
});
