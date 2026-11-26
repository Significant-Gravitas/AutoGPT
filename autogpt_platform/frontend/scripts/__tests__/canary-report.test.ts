// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildReport,
  coloursIn,
  contrastRatio,
  formatReport,
  leaksOf,
  readCanary,
  visibleShadow,
  type ElementFacts,
} from "../ds/canary-report";

const canary = readCanary(
  readFileSync(
    path.resolve(__dirname, "../../src/app/tokens/themes/canary.css"),
    "utf8",
  ),
);

function facts(overrides: Partial<ElementFacts>): ElementFacts {
  return {
    element: "div.card",
    within: "card",
    origin: "ours",
    colours: [],
    radius: null,
    font: null,
    ...overrides,
  };
}

describe("readCanary", () => {
  it("reads every colour triplet and the font families", () => {
    expect(canary.palette).toContainEqual([120, 255, 120]);
    expect(canary.palette).toContainEqual([255, 0, 0]);
    expect(canary.fonts).toEqual(["Papyrus", "Comic Sans MS"]);
  });
});

describe("coloursIn", () => {
  it("reads rgb, rgba, color(srgb) and oklab in shadows and gradients", () => {
    const found = coloursIn(
      "rgba(255, 0, 0, 0.1) 0px 1px 2px 0px, linear-gradient(rgb(0, 90, 0), color(srgb 1 0 0 / 0.5)), oklab(0.627955 0.224863 0.125846)",
    ).map(({ colour }) => colour);
    expect(found).toEqual([
      { rgb: [255, 0, 0], alpha: 0.1 },
      { rgb: [0, 90, 0], alpha: 1 },
      { rgb: [255, 0, 0], alpha: 0.5 },
      { rgb: [255, 0, 0], alpha: 1 },
    ]);
  });
});

describe("visibleShadow", () => {
  it("drops the layers that draw nothing, such as the ring placeholders", () => {
    expect(
      visibleShadow(
        "rgb(255, 255, 255) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(15, 15, 20, 0.04) 0px 1px 1px 0px",
      ),
    ).toBe(" rgba(15, 15, 20, 0.04) 0px 1px 1px 0px");
  });
});

describe("leaksOf", () => {
  it("passes canary colours at any alpha and transparent ones", () => {
    const element = facts({
      colours: [
        { property: "background-color", value: "rgba(170, 255, 170, 0.5)" },
        { property: "border-top-color", value: "rgba(0, 0, 0, 0)" },
        {
          property: "box-shadow",
          value: "rgba(255, 0, 0, 0.1) 0px 1px 2px 0px",
        },
      ],
    });
    expect(leaksOf(element, canary)).toEqual([]);
  });

  it("reports a colour no token gave, with its property", () => {
    const [leak] = leaksOf(
      facts({
        colours: [
          {
            property: "background-image",
            value: "linear-gradient(rgb(139, 92, 246), rgb(0, 90, 0))",
          },
        ],
      }),
      canary,
    );
    expect(leak).toMatchObject({
      kind: "colour",
      property: "background-image",
      value: "rgb(139, 92, 246)",
    });
  });

  it("allows square corners and circles, and tells a pill from a radius", () => {
    const box = (corner: string, width: number, height: number) =>
      leaksOf(
        facts({ radius: { corners: Array(4).fill(corner), width, height } }),
        canary,
      );
    expect(box("0px", 40, 20)).toEqual([]);
    expect(box("9999px", 32, 32)).toEqual([]);
    expect(box("50%", 32, 32)).toEqual([]);
    expect(box("8px", 40, 20)[0]).toMatchObject({
      property: "border-radius",
      value: "8px",
    });
    expect(box("9999px", 80, 32)[0]).toMatchObject({
      property: "border-radius (full)",
    });
  });

  it("exempts a circle only when both radius axes reach half the side", () => {
    const box = (corner: string, width: number, height: number) =>
      leaksOf(
        facts({ radius: { corners: Array(4).fill(corner), width, height } }),
        canary,
      );
    expect(box("8px", 16, 16)).toEqual([]);
    expect(box("8px 8px", 16, 16)).toEqual([]);
    expect(box("8px 1px", 16, 16)[0]).toMatchObject({
      property: "border-radius",
      value: "8px 1px",
    });
    expect(box("50% 10%", 16, 16)[0]).toMatchObject({
      property: "border-radius",
      value: "50% 10%",
    });
  });

  it("reports a rounded box of unknown size, such as an inline pseudo-element", () => {
    const pseudo = (corner: string) =>
      leaksOf(
        facts({
          element: "span::before",
          radius: { corners: Array(4).fill(corner), width: 0, height: 0 },
        }),
        canary,
      );
    expect(pseudo("0px")).toEqual([]);
    expect(pseudo("8px")[0]).toMatchObject({
      kind: "radius",
      value: "8px",
      element: "span::before",
    });
  });

  it("reports text in any font but the canary's", () => {
    expect(leaksOf(facts({ font: "Papyrus, fantasy" }), canary)).toEqual([]);
    expect(
      leaksOf(facts({ font: '"Comic Sans MS", cursive' }), canary),
    ).toEqual([]);
    expect(leaksOf(facts({ font: "Geist" }), canary)[0]).toMatchObject({
      kind: "font",
      value: "Geist",
    });
  });
});

describe("buildReport", () => {
  it("de-duplicates by element and value, counting instances and targets", () => {
    const leak = leaksOf(facts({ font: "Geist" }), canary);
    const report = buildReport([
      { target: "a", leaks: [...leak, ...leak] },
      { target: "b", leaks: leak },
    ]);
    expect(report.totals.font).toEqual({ instances: 3, distinct: 1 });
    expect(report.leaks.font[0]).toMatchObject({
      count: 3,
      targets: ["a", "b"],
    });
    expect(formatReport(report)).toMatch(
      /3x {2}font-family: Geist {2}div\.card \(in card\) \[ours\] {2}a, b/,
    );
  });

  it("keeps our leak apart from the same leak in a library's markup", () => {
    const ours = leaksOf(facts({ font: "Geist" }), canary);
    const theirs = leaksOf(facts({ font: "Geist", origin: "sonner" }), canary);
    const report = buildReport([{ target: "a", leaks: [...ours, ...theirs] }]);
    expect(report.totals.font).toEqual({ instances: 2, distinct: 2 });
    expect(report.leaks.font.map((leak) => leak.origin).sort()).toEqual([
      "ours",
      "sonner",
    ]);
  });
});

describe("contrastRatio", () => {
  it("matches WCAG's figures", () => {
    expect(contrastRatio([0, 0, 0], [255, 255, 255])).toBeCloseTo(21, 5);
    expect(contrastRatio([118, 118, 118], [255, 255, 255])).toBeCloseTo(
      4.54,
      2,
    );
    expect(contrastRatio([255, 255, 255], [118, 118, 118])).toBeCloseTo(
      4.54,
      2,
    );
  });
});
