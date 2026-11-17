import { AUTOPILOT_HEX } from "@/components/molecules/ExpertAvatar/colors";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThinkingIndicator } from "../ThinkingIndicator";

vi.mock("../../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));

afterEach(cleanup);

describe("ThinkingIndicator loader", () => {
  it("renders the stone loader at 16px, in the label's own colour", () => {
    const { container } = render(<ThinkingIndicator active />);
    const svg = container.querySelector("svg");

    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("height")).toBe("16");
    expect(svg?.getAttribute("fill")).toBe("currentColor");
  });

  it("gives the loader the colour of whoever is thinking", () => {
    const { container } = render(
      <ThinkingIndicator active color={AUTOPILOT_HEX} />,
    );

    expect(container.querySelector("svg")?.getAttribute("fill")).toBe(
      AUTOPILOT_HEX,
    );
  });

  it("announces the status politely to screen readers", () => {
    render(<ThinkingIndicator active />);
    const status = screen.getByRole("status");

    expect(status.getAttribute("aria-live")).toBe("polite");
    expect(status.textContent).toBe("Thinking…");
  });
});
