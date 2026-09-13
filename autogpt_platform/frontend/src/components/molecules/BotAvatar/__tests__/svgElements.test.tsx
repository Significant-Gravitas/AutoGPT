import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { STATIC_ELS } from "../svgElements";

describe("static SVG transforms", () => {
  it("preserves authored placement when applying the animation resting transform", () => {
    const svg = renderToStaticMarkup(
      <svg>
        <STATIC_ELS.g
          transform="translate(20 30)"
          animate={{ x: 4, y: 5, rotate: 12, scaleX: 2, scaleY: 3 }}
        >
          <circle r={1} />
        </STATIC_ELS.g>
      </svg>,
    );
    expect(svg).toContain(
      'transform="translate(20 30) translate(4 5) rotate(12) scale(2 3)"',
    );
    expect(svg).not.toContain("animate=");
  });
  it("retains authored transforms for an identity resting frame", () => {
    const svg = renderToStaticMarkup(
      <svg>
        <STATIC_ELS.g transform="rotate(45)" animate={{ x: 0, scale: 1 }} />
      </svg>,
    );
    expect(svg).toContain('transform="rotate(45)"');
  });
  it("emits an animation transform without an authored transform", () => {
    const svg = renderToStaticMarkup(
      <svg>
        <STATIC_ELS.g animate={{ scale: 2 }} />
      </svg>,
    );
    expect(svg).toContain('transform="scale(2 2)"');
  });
});
