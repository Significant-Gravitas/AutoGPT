import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { render, screen, cleanup } from "@testing-library/react";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { afterEach, describe, expect, it } from "vitest";
import { Button } from "./Button";
import { ButtonProps } from "./helpers";

function renderButton(
  props: Partial<ButtonProps> & { children?: React.ReactNode } = {},
) {
  const { children = "Button", ...rest } = props;
  return render(
    <TooltipProvider>
      <Button {...(rest as ButtonProps)}>{children}</Button>
    </TooltipProvider>,
  );
}

afterEach(() => {
  cleanup();
});

describe("Button compact sizes", () => {
  it("renders xs as a 28px rounded-rectangle chip", () => {
    renderButton({ variant: "secondary", size: "xs", children: "Chat" });
    const el = screen.getByRole("button", { name: "Chat" });
    expect(el.className).toContain("h-7");
    expect(el.className).toContain("rounded-md");
    expect(el.className).not.toContain("rounded-full");
  });

  it("softens the outline border at xs", () => {
    renderButton({ variant: "outline", size: "xs", children: "New Pod" });
    const el = screen.getByRole("button", { name: "New Pod" });
    expect(el.className).toContain("border-zinc-200");
    expect(el.className).not.toContain("border-zinc-700");
  });

  it("renders a leadingIcon before the label", () => {
    renderButton({
      size: "xs",
      leadingIcon: PencilEdit02Icon,
      children: "Edit Soul",
    });
    const el = screen.getByRole("button", { name: "Edit Soul" });
    expect(el.querySelector("svg")).not.toBeNull();
    expect(el.querySelector("svg")?.getAttribute("width")).toBe("14");
  });

  it("names icon-only buttons by aria-label", () => {
    renderButton({
      variant: "floating",
      size: "icon-xs",
      leadingIcon: PencilEdit02Icon,
      "aria-label": "Edit workflow",
      children: undefined,
    });
    const el = screen.getByRole("button", { name: "Edit workflow" });
    expect(el.className).toContain("size-7");
    expect(el.className).toContain("bg-white/90");
  });

  it("styles toggle by aria-pressed", () => {
    renderButton({
      variant: "toggle",
      size: "xs",
      "aria-pressed": true,
      children: "Needs review",
    });
    const el = screen.getByRole("button", {
      name: "Needs review",
      pressed: true,
    });
    expect(el.className).toContain("aria-pressed:bg-zinc-100");
  });
});
