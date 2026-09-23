import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { markdownRenderer } from "../MarkdownRenderer";

function renderMarkdown(markdown: string) {
  return render(
    <>
      {markdownRenderer.render(markdown, {
        mimeType: "text/markdown",
        filename: "notes.md",
      })}
    </>,
  ).container;
}

// KaTeX keeps the TeX source in a MathML annotation, so prose assertions have to
// read the document with the rendered formulas taken out.
function proseOf(container: HTMLElement): string {
  const copy = container.cloneNode(true) as HTMLElement;
  copy.querySelectorAll(".katex").forEach((node) => node.remove());
  return copy.textContent ?? "";
}

function texOf(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll('annotation[encoding="application/x-tex"]'),
  ).map((node) => node.textContent ?? "");
}

describe("MarkdownRenderer math", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders inline $…$ math through KaTeX", () => {
    const container = renderMarkdown(
      "The condition is $\\deg(\\text{num}) \\geq \\deg(\\text{den})$ here.",
    );

    expect(container.querySelector(".katex")).toBeTruthy();
    expect(texOf(container)).toEqual([
      "\\deg(\\text{num}) \\geq \\deg(\\text{den})",
    ]);
    expect(proseOf(container)).toBe("The condition is  here.");
  });

  it("still renders display $$…$$ math", () => {
    const container = renderMarkdown("Result:\n\n$$\n\\frac{1}{s+1}\n$$\n");

    expect(container.querySelector(".katex-display")).toBeTruthy();
    expect(texOf(container)).toEqual(["\\frac{1}{s+1}"]);
  });

  it("leaves two currency amounts in one sentence as literal text", () => {
    const container = renderMarkdown("$5 and $10 buys lunch.");

    expect(container.querySelector(".katex")).toBeNull();
    expect(proseOf(container)).toBe("$5 and $10 buys lunch.");
  });

  it("renders the math and keeps the price in a sentence holding both", () => {
    const container = renderMarkdown("It costs $5 to compute $(s+1)^{2}$ now.");

    expect(texOf(container)).toEqual(["(s+1)^{2}"]);
    expect(proseOf(container)).toBe("It costs $5 to compute  now.");
  });

  it("keeps digit-leading math with LaTeX syntax as math", () => {
    const container = renderMarkdown("Growth follows $5x^2$ exactly.");

    expect(texOf(container)).toEqual(["5x^2"]);
    expect(proseOf(container)).toBe("Growth follows  exactly.");
  });

  it("leaves dollars inside a fenced code block untouched", () => {
    const container = renderMarkdown("```bash\necho $5 and $10\n```\n");

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector("code")?.textContent).toContain(
      "echo $5 and $10",
    );
  });
});
