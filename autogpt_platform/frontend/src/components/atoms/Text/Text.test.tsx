import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { tones, type Tone } from "./helpers";
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
    expect(el.className).toContain("text-foreground-tertiary");
    expect(el.className).not.toContain("text-foreground-base");
  });

  it("keeps the variant's foreground-base when no tone is given", () => {
    render(<Text variant="body">Plain</Text>);
    expect(screen.getByText("Plain").className).toContain(
      "text-foreground-base",
    );
  });

  it.each(Object.entries(tones))(
    "renders the %s tone with its role class only",
    (tone, roleClass) => {
      render(
        <Text variant="body" tone={tone as Tone}>
          Toned
        </Text>,
      );
      const classes = screen.getByText("Toned").className.split(" ");
      expect(classes).toContain(roleClass);
      expect(classes).not.toContain("text-foreground-base");
    },
  );

  it("renders eyebrow as a 12px uppercase span", () => {
    render(<Text variant="eyebrow">Needs you</Text>);
    const el = screen.getByText("Needs you");
    expect(el.tagName).toBe("SPAN");
    expect(el.className).toContain("uppercase");
    expect(el.className).toContain("text-caption");
  });
});
