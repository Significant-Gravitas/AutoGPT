import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Text } from "./Text";

afterEach(() => {
  cleanup();
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
