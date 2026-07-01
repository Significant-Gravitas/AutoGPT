import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Text } from "./Text";

afterEach(() => {
  cleanup();
});

describe("Text unmask prop", () => {
  it("applies sentry-unmask class by default", () => {
    render(<Text variant="body">Static label</Text>);
    const el = screen.getByText("Static label");
    expect(el.className).toContain("sentry-unmask");
  });

  it("omits sentry-unmask class when unmask is false", () => {
    render(
      <Text variant="body" unmask={false}>
        Dynamic content
      </Text>,
    );
    const el = screen.getByText("Dynamic content");
    expect(el.className).not.toContain("sentry-unmask");
  });

  it("applies sentry-unmask when unmask is explicitly true", () => {
    render(
      <Text variant="body" unmask={true}>
        Explicit unmask
      </Text>,
    );
    const el = screen.getByText("Explicit unmask");
    expect(el.className).toContain("sentry-unmask");
  });

  it("preserves custom className alongside sentry-unmask", () => {
    render(
      <Text variant="body" className="custom-class">
        With custom class
      </Text>,
    );
    const el = screen.getByText("With custom class");
    expect(el.className).toContain("sentry-unmask");
    expect(el.className).toContain("custom-class");
  });

  it("renders correct element for heading variants with unmask", () => {
    render(<Text variant="h1">Heading</Text>);
    const el = screen.getByText("Heading");
    expect(el.tagName).toBe("H1");
    expect(el.className).toContain("sentry-unmask");
  });

  it("renders correct element for body variant with unmask disabled", () => {
    render(
      <Text variant="body" unmask={false}>
        Body text
      </Text>,
    );
    const el = screen.getByText("Body text");
    expect(el.tagName).toBe("P");
    expect(el.className).not.toContain("sentry-unmask");
  });
});
