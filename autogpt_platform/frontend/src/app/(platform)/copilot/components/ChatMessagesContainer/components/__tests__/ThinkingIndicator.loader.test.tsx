import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThinkingIndicator } from "../ThinkingIndicator";

vi.mock("../../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));

// PixelGridLoader renders 9 cells in a hidden grid; the drawn loader is an svg.
function loaders(container: HTMLElement) {
  return {
    pixelGrid: container.querySelector("span[aria-hidden] > span"),
    drawn: container.querySelector("svg"),
  };
}

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
});

describe("ThinkingIndicator loader selection", () => {
  it("renders the pixel grid when the experiment flag is off", () => {
    vi.stubEnv("NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER", "false");
    const { container } = render(<ThinkingIndicator active />);
    const { pixelGrid, drawn } = loaders(container);

    expect(pixelGrid).not.toBeNull();
    expect(drawn).toBeNull();
  });

  it("renders the drawn line loader at 16px when the flag is on", () => {
    vi.stubEnv("NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER", "true");
    const { container } = render(<ThinkingIndicator active />);
    const { pixelGrid, drawn } = loaders(container);

    expect(drawn).not.toBeNull();
    expect(drawn?.getAttribute("height")).toBe("16");
    expect(pixelGrid).toBeNull();
  });

  it("treats any value other than 'true' as off", () => {
    vi.stubEnv("NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER", "1");
    const { container } = render(<ThinkingIndicator active />);

    expect(loaders(container).drawn).toBeNull();
  });

  it("lets the loader prop override the flag (styleguide)", () => {
    const { container } = render(<ThinkingIndicator active loader="drawn" />);
    expect(loaders(container).drawn).not.toBeNull();
  });

  it("keeps the screen reader output the same either way", () => {
    for (const loader of ["pixel", "drawn"] as const) {
      const { unmount } = render(<ThinkingIndicator active loader={loader} />);
      const status = screen.getByRole("status");
      expect(status.getAttribute("aria-live")).toBe("polite");
      expect(status.textContent).toBe("Thinking…");
      unmount();
    }
  });
});
