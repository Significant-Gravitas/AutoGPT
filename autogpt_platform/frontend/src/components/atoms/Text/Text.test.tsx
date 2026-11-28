import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { tones, type Tone } from "@/components/styles/tones";
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

  it.each(Object.entries(tones).filter(([tone]) => tone !== "base"))(
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

describe("Text modifier props", () => {
  it("replaces the variant's weight", () => {
    render(
      <Text variant="h4" weight="semibold">
        Weighted
      </Text>,
    );
    const classes = screen.getByText("Weighted").className.split(" ");
    expect(classes).toContain("font-semibold");
    expect(classes).not.toContain("font-medium");
  });

  it("swaps the family for mono and keeps the size", () => {
    render(
      <Text variant="small" mono>
        abc123
      </Text>,
    );
    const classes = screen.getByText("abc123").className.split(" ");
    expect(classes).toContain("font-mono");
    expect(classes).not.toContain("font-sans");
    expect(classes).toContain("text-caption");
  });

  it.each([
    [{ tabular: true }, "tabular-nums"],
    [{ italic: true }, "italic"],
    [{ decoration: "line-through" as const }, "line-through"],
    [{ decoration: "none" as const }, "no-underline"],
    [{ transform: "uppercase" as const }, "uppercase"],
    [{ transform: "none" as const }, "normal-case"],
    [{ align: "center" as const }, "text-center"],
    [{ tracking: "caps" as const }, "tracking-caps"],
    [{ leading: "5" as const }, "leading-5"],
    [{ leading: "inherit" as const }, "leading-inherit"],
    [{ leading: "snug" as const }, "leading-snug"],
    [{ motion: "colors" as const }, "transition-colors"],
  ])("renders %o as %s", (props, expected) => {
    render(
      <Text variant="body" {...props}>
        Modified
      </Text>,
    );
    expect(screen.getByText("Modified").className.split(" ")).toContain(
      expected,
    );
  });

  it("replaces the variant's line height", () => {
    render(
      <Text variant="h5" leading="8">
        Leading
      </Text>,
    );
    const classes = screen.getByText("Leading").className.split(" ");
    expect(classes).toContain("leading-8");
    expect(classes).not.toContain("leading-6");
  });

  it("steps up below md with narrowSize and keeps the base size", () => {
    render(
      <Text variant="small" leading="4" narrowSize="body">
        Narrow
      </Text>,
    );
    const classes = screen.getByText("Narrow").className.split(" ");
    expect(classes).toEqual(
      expect.arrayContaining([
        "text-caption",
        "leading-4",
        "max-md:text-body",
        "max-md:leading-5",
      ]),
    );
  });

  it("lets className win for layout next to the props", () => {
    render(
      <Text variant="body" tone="muted" tabular className="mt-2 truncate">
        Layout
      </Text>,
    );
    const classes = screen.getByText("Layout").className.split(" ");
    expect(classes).toEqual(
      expect.arrayContaining([
        "text-foreground-tertiary",
        "tabular-nums",
        "mt-2",
        "truncate",
      ]),
    );
  });
});
