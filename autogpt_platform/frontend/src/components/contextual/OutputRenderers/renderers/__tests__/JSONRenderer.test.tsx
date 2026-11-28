import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { jsonRenderer } from "../JSONRenderer";

function renderJSON(value: unknown) {
  return render(<>{jsonRenderer.render(value, {})}</>);
}

describe("jsonRenderer", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows every primitive the way it reads in JSON", () => {
    renderJSON({ enabled: true, count: 42, name: "Otto", missing: null });

    expect(screen.getByText("true")).toBeDefined();
    expect(screen.getByText("42")).toBeDefined();
    expect(screen.getByText('"Otto"')).toBeDefined();
    expect(screen.getByText("null")).toBeDefined();
  });

  it("labels each key of an object", () => {
    renderJSON({ enabled: false });

    expect(screen.getByText(`"enabled":`)).toBeDefined();
    expect(screen.getByText("false")).toBeDefined();
  });

  it.each([
    ['{"name": "Otto"}', true],
    ["[1, 2]", true],
    ["1e5", false],
    ["1.50", false],
    ["true", false],
    ["null", false],
    ['"Otto"', false],
    ["not JSON", false],
  ])("takes the text %s as JSON: %s", (text, isJSON) => {
    expect(jsonRenderer.canRender(text)).toBe(isJSON);
  });

  it("takes a value its metadata calls JSON, and any object", () => {
    expect(jsonRenderer.canRender("1e5", { type: "json" })).toBe(true);
    expect(jsonRenderer.canRender({ count: 1 })).toBe(true);
    expect(jsonRenderer.canRender([1])).toBe(true);
    expect(jsonRenderer.canRender(42)).toBe(false);
  });

  it("colours strings orange and quiet values in the tertiary role", () => {
    renderJSON({
      name: "Otto",
      tags: [],
      meta: {},
      missing: undefined,
      big: BigInt(10),
    });

    expect(screen.getByText('"Otto"').className).toContain(
      "text-orange-foreground",
    );
    for (const quiet of ["[]", "{}", "undefined", "10"])
      expect(screen.getByText(quiet).className).toContain(
        "text-foreground-tertiary",
      );
  });
});
