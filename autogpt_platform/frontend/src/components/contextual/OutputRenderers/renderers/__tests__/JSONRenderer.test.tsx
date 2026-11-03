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
});
