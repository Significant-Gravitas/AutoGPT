import { EXPERT_PALETTE } from "@/components/molecules/ExpertAvatar/helpers";
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StoneStackLoader } from "../StoneStackLoader";
import { HEAD_STONES } from "../stoneStackShapes";

const motion = vi.hoisted(() => ({ reduced: false }));
vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => motion.reduced,
}));

function renderLoader(
  props: React.ComponentProps<typeof StoneStackLoader> = {},
) {
  const { container } = render(<StoneStackLoader {...props} />);
  const svg = container.firstElementChild as SVGSVGElement;
  const animations = Array.from(svg.querySelectorAll("*")).filter((node) =>
    ["animate", "animateTransform"].includes(node.localName),
  );
  const morph = animations.find(
    (node) => node.getAttribute("attributeName") === "d",
  );
  return { svg, animations, morph, head: morph?.parentElement ?? null };
}

/** `M 1 2 Q 3 4 5 6` -> `MQ`: what a browser needs to match to morph. */
function commandsOf(d: string) {
  return d.replace(/[^a-zA-Z]/g, "");
}

afterEach(() => {
  cleanup();
  motion.reduced = false;
});

describe("StoneStackLoader", () => {
  it("renders a hidden 20px svg by default", () => {
    const { svg } = renderLoader();

    expect(svg.tagName.toLowerCase()).toBe("svg");
    expect(svg.getAttribute("aria-hidden")).toBe("true");
    expect(svg.getAttribute("width")).toBe("20");
    expect(svg.getAttribute("height")).toBe("20");
    expect(svg.getAttribute("viewBox")).toBe("0 0 24 24");
  });

  it("takes the text colour unless it is given a colour", () => {
    expect(renderLoader().svg.getAttribute("fill")).toBe("currentColor");

    const lavender = EXPERT_PALETTE.otto.hex;
    expect(renderLoader({ color: lavender }).svg.getAttribute("fill")).toBe(
      lavender,
    );
  });

  it("scales with size and forwards className", () => {
    const { svg } = renderLoader({ size: 16, className: "text-red" });

    expect(svg.getAttribute("width")).toBe("16");
    expect(svg.getAttribute("height")).toBe("16");
    expect(svg.getAttribute("class")).toContain("text-red");
  });

  it("starts on the first head and morphs through every head back to it", () => {
    const { morph, head } = renderLoader();
    const values = morph?.getAttribute("values")?.split(";") ?? [];

    expect(head?.getAttribute("d")).toBe(HEAD_STONES[0].d);
    expect(values[0]).toBe(HEAD_STONES[0].d);
    expect(values.at(-1)).toBe(HEAD_STONES[0].d);
    HEAD_STONES.forEach((stone) => expect(values).toContain(stone.d));
  });

  it("gives every animation as many key times as values", () => {
    const { animations } = renderLoader();

    expect(animations.length).toBeGreaterThan(0);
    animations.forEach((animation) => {
      const count = (name: string) =>
        animation.getAttribute(name)?.split(";").length;
      expect(count("keyTimes")).toBe(count("values"));
      expect(count("keySplines")).toBe((count("values") ?? 0) - 1);
    });
  });

  it("draws every head with the same path commands, so they can morph", () => {
    const commands = commandsOf(HEAD_STONES[0].d);

    HEAD_STONES.forEach((stone) =>
      expect(commandsOf(stone.d), stone.name).toBe(commands),
    );
  });

  it("stands still under reduced motion", () => {
    motion.reduced = true;
    const { svg, animations } = renderLoader();

    expect(animations).toHaveLength(0);
    expect(svg.querySelectorAll("path").length).toBeGreaterThan(0);
  });
});
