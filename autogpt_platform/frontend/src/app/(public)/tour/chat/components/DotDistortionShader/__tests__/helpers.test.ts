import { afterEach, describe, expect, it } from "vitest";
import { resolveCanvasColour } from "../helpers";

describe("resolveCanvasColour", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  function elementWith(style: string) {
    const element = document.createElement("div");
    element.setAttribute("style", style);
    document.body.appendChild(element);
    return element;
  }

  it("wraps a colour role's RGB triplet in rgb()", () => {
    const element = elementWith("--color-foreground-base: 20 20 20");
    expect(resolveCanvasColour("var(--color-foreground-base)", element)).toBe(
      "rgb(20 20 20)",
    );
  });

  it("passes a variable that already holds a colour through", () => {
    const element = elementWith("--dot: #336699");
    expect(resolveCanvasColour("var(--dot)", element)).toBe("#336699");
  });

  it("leaves a plain colour alone", () => {
    expect(resolveCanvasColour("rgb(1 2 3)", elementWith(""))).toBe(
      "rgb(1 2 3)",
    );
  });
});
