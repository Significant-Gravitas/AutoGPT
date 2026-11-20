import { Tick02Icon } from "@hugeicons/core-free-icons";
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { tones } from "@/components/styles/tones";
import { Icon } from "./Icon";

afterEach(() => {
  cleanup();
});

function classesOf(container: HTMLElement) {
  return (container.querySelector("svg")?.getAttribute("class") ?? "").split(
    " ",
  );
}

describe("Icon", () => {
  it("renders 1em with the design-system stroke by default", () => {
    const { container } = render(<Icon icon={Tick02Icon} />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("width")).toBe("1em");
    expect(svg?.getAttribute("stroke-width")).toBe("2");
  });

  it.each(Object.entries(tones))("renders the %s tone as %s", (tone, cls) => {
    const { container } = render(
      <Icon icon={Tick02Icon} tone={tone as keyof typeof tones} />,
    );
    expect(classesOf(container)).toContain(cls);
  });

  it("spins and stops under reduced motion", () => {
    const { container } = render(<Icon icon={Tick02Icon} spin />);
    expect(classesOf(container)).toEqual(
      expect.arrayContaining(["animate-spin", "motion-reduce:animate-none"]),
    );
  });

  it.each([
    ["transform", "transition-transform"],
    ["colors", "transition-colors"],
    ["opacity", "transition-opacity"],
  ] as const)("animates %s changes", (motion, cls) => {
    const { container } = render(<Icon icon={Tick02Icon} motion={motion} />);
    expect(classesOf(container)).toContain(cls);
  });

  it("lets a className colour beat the tone", () => {
    const { container } = render(
      <Icon icon={Tick02Icon} tone="muted" className="size-4 text-error" />,
    );
    const classes = classesOf(container);
    expect(classes).toEqual(expect.arrayContaining(["size-4", "text-error"]));
    expect(classes).not.toContain("text-foreground-tertiary");
  });

  it("passes className through untouched without variant props", () => {
    const conflicting = ["h-4", "w-4", "size-5"].join(" ");
    const { container } = render(
      <Icon icon={Tick02Icon} className={conflicting} />,
    );
    expect(container.querySelector("svg")?.getAttribute("class")).toBe(
      conflicting,
    );
  });
});
