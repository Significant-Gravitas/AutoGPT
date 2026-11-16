import { render, screen } from "@/tests/integrations/test-utils";
import { useRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TextGenerateEffect } from "../TextGenerateEffect";

const animate = vi.fn();

// The scope keeps one ref object across renders, as Motion's does, so the
// effect sees `scope.current` go from null to the element.
vi.mock("motion/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("motion/react")>();
  return {
    ...actual,
    useAnimate: () => [useRef(null), animate],
  };
});

afterEach(() => {
  animate.mockClear();
});

describe("TextGenerateEffect", () => {
  it("renders every word hidden, then reveals them on mount", () => {
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

  it("reveals once more on the first re-render, when the scope is attached, and not after", () => {
    const { rerender } = render(
      <TextGenerateEffect words="One line" filter={false} duration={0.6} />,
    );
    rerender(
      <TextGenerateEffect words="One line" filter={false} duration={0.6} />,
    );
    expect(animate).toHaveBeenCalledTimes(2);

    rerender(
      <TextGenerateEffect words="One line" filter={false} duration={0.6} />,
    );
    expect(animate).toHaveBeenCalledTimes(2);
    expect(animate).toHaveBeenLastCalledWith(
      "span",
      { opacity: 1, filter: "none" },
      expect.objectContaining({ duration: 0.6 }),
    );
  });

  it("uses the props of the render it runs after, and ignores later prop changes", () => {
    const { rerender } = render(
      <TextGenerateEffect words="First words" duration={0.5} />,
    );
    rerender(<TextGenerateEffect words="First words" duration={0.7} />);
    expect(animate).toHaveBeenCalledTimes(2);
    expect(animate).toHaveBeenLastCalledWith(
      "span",
      { opacity: 1, filter: "blur(0px)" },
      expect.objectContaining({ duration: 0.7 }),
    );

    rerender(<TextGenerateEffect words="Other words" duration={0.9} />);
    rerender(<TextGenerateEffect words="Other words" duration={0.9} />);
    expect(animate).toHaveBeenCalledTimes(2);
  });
});
