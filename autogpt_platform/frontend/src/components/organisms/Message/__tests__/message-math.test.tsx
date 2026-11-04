import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageResponse } from "../Message";

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

async function renderReply(markdown: string) {
  const { container } = render(<MessageResponse>{markdown}</MessageResponse>);
  await waitFor(() => expect(container.textContent).not.toBe(""));
  return container;
}

describe("MessageResponse math", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders inline $…$ math through KaTeX", async () => {
    const container = await renderReply(
      "The condition is $\\deg(\\text{num}) \\geq \\deg(\\text{den})$ here.",
    );

    await waitFor(() => expect(container.querySelector(".katex")).toBeTruthy());
    expect(texOf(container)).toEqual([
      "\\deg(\\text{num}) \\geq \\deg(\\text{den})",
    ]);
    expect(proseOf(container)).toBe("The condition is  here.");
  });

  it("still renders display $$…$$ math", async () => {
    const container = await renderReply("Result:\n\n$$\n\\frac{1}{s+1}\n$$\n");

    await waitFor(() =>
      expect(container.querySelector(".katex-display")).toBeTruthy(),
    );
    expect(texOf(container)).toEqual(["\\frac{1}{s+1}"]);
  });

  it("renders a formula holding non-Latin text without a console warning", async () => {
    // Console capture filed each of KaTeX's strict-mode warnings as a Sentry
    // issue, fourteen for one Bengali reader's replies (BUILDER-94S…956).
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const container = await renderReply("ধরা যাক $ভ + ২ = x$ তাহলে");

    await waitFor(() => expect(texOf(container)).toEqual(["ভ + ২ = x"]));
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("leaves two currency amounts in one sentence as literal text", async () => {
    const container = await renderReply("$5 and $10 buys lunch.");

    expect(container.querySelector(".katex")).toBeNull();
    expect(proseOf(container)).toBe("$5 and $10 buys lunch.");
  });

  it("renders the math and keeps the price in a sentence holding both", async () => {
    const container = await renderReply(
      "It costs $5 to compute $(s+1)^{2}$ now.",
    );

    await waitFor(() => expect(texOf(container)).toEqual(["(s+1)^{2}"]));
    expect(proseOf(container)).toBe("It costs $5 to compute  now.");
  });

  it("leaves dollars inside a fenced code block untouched", async () => {
    const container = await renderReply("```bash\necho $5 and $10\n```\n");

    expect(container.querySelector(".katex")).toBeNull();
    await waitFor(() =>
      expect(container.textContent).toContain("echo $5 and $10"),
    );
  });

  it.each([
    [
      "a pasted shell command",
      '$ for f in $(ls logs); do echo "== $f ($(wc -l < $f))"; done',
    ],
    ["shell variables in prose", "Both $HOME and $PATH are set."],
    [
      "two variables joined by a separator",
      "Run as $USER:$GROUP from $HOME/$DIR.",
    ],
    ["a path list", "Set PATH=$HOME/bin:$PATH first."],
    ["braced variables", "Join ${A}:${B} here."],
    ["two adjacent variables", "Print $HOME$PATH now."],
  ])("leaves %s as literal text", async (_, markdown) => {
    const container = await renderReply(markdown);

    expect(container.querySelector(".katex")).toBeNull();
    expect(proseOf(container)).toBe(markdown);
  });

  it.each([
    [
      "a fenced block in a blockquote",
      "> ~~~bash\n> echo $HOME\n> ~~~",
      "pre",
      "echo $HOME",
    ],
    ["an HTML block", "<div>echo $HOME</div>", "div", "echo $HOME"],
    [
      "an HTML comment block holding a browser comment end",
      "<!-- a --!>\necho $HOME\n-->",
      "div",
      "<!-- a --!> echo $HOME -->",
    ],
    [
      "a code span that wraps a line",
      "Run `echo $HOME\nand $PATH` now.",
      "code",
      "echo $HOME and $PATH",
    ],
    [
      "a code span wrapping past an indented backtick line",
      "Run `echo $HOME\n    ```\nand $PATH` now.",
      "code",
      "echo $HOME ``` and $PATH",
    ],
    [
      "a fence opened on a list item's marker line",
      "- ```bash\n  echo $HOME\n  ```",
      "pre",
      "echo $HOME",
    ],
    [
      "a code span holding a longer backtick run",
      "Run ``a ```b` $HOME`` now.",
      "code",
      "a ```b` $HOME",
    ],
    [
      "indented code after a heading that ends a list",
      "- item\n# Heading\n    echo $HOME",
      "code",
      "echo $HOME",
    ],
    [
      "a fence closed in Windows line endings",
      "```\r\necho $HOME\r\n```\r\nIt costs $5 and $10.",
      "pre",
      "echo $HOME",
    ],
  ])("leaves dollars in %s unescaped", async (_, markdown, selector, text) => {
    const container = await renderReply(markdown);

    await waitFor(() =>
      expect(
        container
          .querySelector(selector)
          ?.textContent?.replace(/\s+/g, " ")
          .trim(),
      ).toBe(text),
    );
    expect(container.textContent).not.toContain("\\$");
    expect(container.querySelector(".katex")).toBeNull();
  });

  it.each([
    ["a list item", "Use ` here\n- then $HOME and $PATH` go"],
    ["a heading", "Use ` here\n# Then $HOME and $PATH` go"],
    ["a blockquoted HTML block", "> <pre>\nThen $HOME and $PATH go"],
    ["a sibling list item", "- item\n  ```\n  code\n- Then $HOME and $PATH go"],
  ])("keeps shell text literal after %s ends a block", async (_, markdown) => {
    const container = await renderReply(markdown);

    await waitFor(() =>
      expect(proseOf(container)).toContain("$HOME and $PATH"),
    );
    expect(container.querySelector(".katex")).toBeNull();
  });

  it("keeps the dollar in an inline HTML link's address", async () => {
    const container = await renderReply('See <a href="/docs/$HOME">docs</a>.');

    await waitFor(() =>
      expect(
        container
          .querySelector('[data-streamdown="link"]')
          ?.getAttribute("href"),
      ).toBe("/docs/$HOME"),
    );
  });

  it("closes no formula on a dollar inside inline code", async () => {
    const container = await renderReply("Use $HOME and `$` here.");

    await waitFor(() =>
      expect(container.querySelector("code")?.textContent).toBe("$"),
    );
    expect(container.querySelector(".katex")).toBeNull();
  });

  it("pairs no dollars across a line break", async () => {
    const container = await renderReply(
      "Costs $x on one line\nand y$ on the next.",
    );

    await waitFor(() => expect(proseOf(container)).toContain("$x on one line"));
    expect(container.querySelector(".katex")).toBeNull();
  });

  it("renders a one-letter formula followed by a suffix", async () => {
    const container = await renderReply("Take the $n$th term.");

    await waitFor(() => expect(texOf(container)).toEqual(["n"]));
  });

  it("pairs no math across a dotless URL", async () => {
    const container = await renderReply(
      "Open https://localhost/$HOME and $x$ now.",
    );

    await waitFor(() => expect(texOf(container)).toEqual(["x"]));
    expect(proseOf(container)).toContain("https://localhost/$HOME and");
  });

  it("escapes prices beside a lone backtick run that a fenced block follows", async () => {
    const container = await renderReply(
      "Type ``` to pay $5 and $10.\n```bash\necho hi\n```",
    );

    await waitFor(() => expect(container.querySelector("pre")).toBeTruthy());
    expect(container.querySelector(".katex")).toBeNull();
    expect(proseOf(container)).toContain("to pay $5 and $10.");
  });

  it.each([
    ["an autolink", "<https://example.org/odata/$metadata>"],
    ["a bare URL", "See https://example.org/odata/$metadata now."],
  ])("keeps the dollar in %s's address", async (_, markdown) => {
    const container = await renderReply(markdown);

    // Link safety renders a link as a button labelled with its address.
    await waitFor(() =>
      expect(
        container.querySelector('[data-streamdown="link"]')?.textContent,
      ).toBe("https://example.org/odata/$metadata"),
    );
  });

  it("renders math whose \\left delimiter has no partner", async () => {
    const container = await renderReply("Take $\\left( x \\right.$ here.");

    await waitFor(() =>
      expect(texOf(container)).toEqual(["\\left( x \\right."]),
    );
  });
});
