import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThinkingIndicator } from "../ThinkingIndicator";

vi.mock("../../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));

afterEach(cleanup);

describe("ThinkingIndicator loader", () => {
  it("renders the drawn line loader at 16px", () => {
    const { container } = render(<ThinkingIndicator active />);
    const svg = container.querySelector("svg");

    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("height")).toBe("16");
  });

  it("announces the status politely to screen readers", () => {
    render(<ThinkingIndicator active />);
    const status = screen.getByRole("status");

    expect(status.getAttribute("aria-live")).toBe("polite");
    expect(status.textContent).toBe("Thinking…");
  });
});
