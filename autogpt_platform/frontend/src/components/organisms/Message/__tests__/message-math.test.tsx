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
  ])("leaves %s as literal text", async (_, markdown) => {
    const container = await renderReply(markdown);

    expect(container.querySelector(".katex")).toBeNull();
    expect(proseOf(container)).toBe(markdown);
  });

  it.each([
    ["a fenced block in a blockquote", "> ~~~bash\n> echo $HOME\n> ~~~", "pre"],
    ["an HTML block", "<div>echo $HOME</div>", "div"],
    [
      "a code span that wraps a line",
      "Run `echo $HOME\nand $PATH` now.",
      "code",
    ],
  ])("leaves dollars in %s unescaped", async (_, markdown, selector) => {
    const container = await renderReply(markdown);

    await waitFor(() =>
      expect(container.querySelector(selector)?.textContent).toMatch(
        /echo \$HOME/,
      ),
    );
    expect(container.textContent).not.toContain("\\$");
  });

  it("escapes prices beside a lone backtick that a fenced block follows", async () => {
    const container = await renderReply(
      "Type ` to pay $5 and $10.\n```bash\necho hi\n```",
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
