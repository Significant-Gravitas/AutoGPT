import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TextGenerateEffect } from "../TextGenerateEffect";

const animate = vi.fn();

vi.mock("motion/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("motion/react")>();
  return {
    ...actual,
    useAnimate: () => [{ current: null }, animate],
  };
});

afterEach(() => {
  animate.mockClear();
});

describe("TextGenerateEffect", () => {
  it("renders every word hidden, then reveals them once on mount", () => {
    render(<TextGenerateEffect words="Tell me what slows you down" />);

    for (const word of ["Tell", "me", "what", "slows", "you", "down"]) {
      expect(screen.getByText(word)).toBeDefined();
    }
    expect(animate).toHaveBeenCalledTimes(1);
    expect(animate).toHaveBeenCalledWith(
      "span",
      { opacity: 1, filter: "blur(0px)" },
      expect.objectContaining({ duration: 0.5 }),
    );
  });

  it("does not reveal again when it re-renders", () => {
    const { rerender } = render(
      <TextGenerateEffect words="One line" filter={false} duration={0.6} />,
    );
    rerender(
      <TextGenerateEffect words="One line" filter={false} duration={0.6} />,
    );

    expect(animate).toHaveBeenCalledTimes(1);
    expect(animate).toHaveBeenCalledWith(
      "span",
      { opacity: 1, filter: "none" },
      expect.objectContaining({ duration: 0.6 }),
    );
  });
});
