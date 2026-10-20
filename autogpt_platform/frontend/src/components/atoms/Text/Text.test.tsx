import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Text } from "./Text";

afterEach(() => {
  cleanup();
});

describe("Text in session replay", () => {
  it("is masked by default", () => {
    render(<Text variant="body">Account name</Text>);
    const el = screen.getByText("Account name");
    expect(el.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
  });

  it("is masked by default for headings too", () => {
    render(<Text variant="h1">Heading</Text>);
    const el = screen.getByText("Heading");
    expect(el.tagName).toBe("H1");
    expect(el.className).not.toContain("sentry-unmask");
  });

  it("opts into unmasking only when asked", () => {
    render(
      <Text variant="body" unmask className="custom-class">
        Static label
      </Text>,
    );
    const el = screen.getByText("Static label");
    expect(el.className).toContain("sentry-unmask");
    expect(el.className).toContain("custom-class");
  });
});

describe("Text tone and eyebrow", () => {
  it("applies the tone colour over the variant default", () => {
    render(
      <Text variant="small" tone="muted">
        Meta
      </Text>,
    );
    const el = screen.getByText("Meta");
    expect(el.className).toContain("text-zinc-500");
    expect(el.className).not.toContain("text-black");
  });

  it("keeps black when no tone is given", () => {
    render(<Text variant="body">Plain</Text>);
    expect(screen.getByText("Plain").className).toContain("text-black");
  });

  it("renders eyebrow as a 12px uppercase span", () => {
    render(<Text variant="eyebrow">Needs you</Text>);
    const el = screen.getByText("Needs you");
    expect(el.tagName).toBe("SPAN");
    expect(el.className).toContain("uppercase");
    expect(el.className).toContain("text-[0.75rem]");
  });
});
