import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MessageResponse } from "../message";

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
});
