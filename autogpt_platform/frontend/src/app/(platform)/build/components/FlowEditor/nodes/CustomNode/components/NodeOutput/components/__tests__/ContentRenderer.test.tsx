import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { TextRenderer } from "../ContentRenderer";

describe("TextRenderer", () => {
  it("shows short text as is on the muted surface", () => {
    render(<TextRenderer value="hello" />);
    const box = screen.getByText("hello");
    expect(box.tagName).toBe("DIV");
    expect(box.className.split(" ")).toEqual(
      expect.arrayContaining(["text-caption", "leading-4", "text-inherit"]),
    );
    expect(box.parentElement?.className.split(" ")).toContain(
      "bg-surface-muted",
    );
  });

  it("truncates text past the limit", () => {
    render(<TextRenderer value={"a".repeat(12)} truncateLengthLimit={5} />);
    expect(screen.getByText("aaaaa...")).toBeDefined();
  });

  it("pretty-prints objects", () => {
    const { container } = render(<TextRenderer value={{ answer: 42 }} />);
    expect(container.querySelector(".bg-surface-muted")?.textContent).toBe(
      '{\n  "answer": 42\n}',
    );
  });
});
