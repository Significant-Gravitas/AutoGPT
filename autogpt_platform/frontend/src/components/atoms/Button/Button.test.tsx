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
    expect(el.className).toContain("border-border-base");
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
    expect(el.className).toContain("bg-surface/90");
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
    expect(el.className).toContain("aria-pressed:bg-surface-hover");
  });
});

describe("Button legacy variants", () => {
  it("renders a plain button with the children as they are", () => {
    render(
      <Button variant="legacy-outline" className="space-x-3">
        <span>One</span>
        <span>Two</span>
      </Button>,
    );
    const el = screen.getByRole("button", { name: "One Two" });
    expect(el.children).toHaveLength(2);
    expect(el.querySelector(".contents")).toBeNull();
    expect(el.className).toContain("border-neutral-200");
    expect(el.className).toContain("h-9 px-4 py-2");
    expect(el.className).toContain("space-x-3");
  });

  it("applies the size, and lets the caller's classes win", () => {
    render(
      <Button variant="legacy-default" size="legacy-sm" className="text-xs">
        Save
      </Button>,
    );
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("h-8");
    expect(el.className).toContain("text-xs");
    expect(el.className).not.toContain("text-caption");
    expect(el.className).not.toContain("leading-4");
  });

  it("forwards the ref and the caller's props", () => {
    let node: HTMLButtonElement | null = null;
    render(
      <Button
        variant="legacy-ghost"
        ref={(el) => {
          node = el;
        }}
        disabled
        data-testid="ghost"
      >
        Ghost
      </Button>,
    );
    const el = screen.getByTestId("ghost");
    expect(node).toBe(el);
    expect(el).toHaveProperty("disabled", true);
  });

  it("adds no slot of its own, like the old component, and keeps the caller's", () => {
    render(
      <>
        <Button variant="legacy-outline">Plain</Button>
        <Button variant="legacy-ghost" data-slot="date-input-trigger">
          Pick
        </Button>
      </>,
    );
    expect(
      screen.getByRole("button", { name: "Plain" }).hasAttribute("data-slot"),
    ).toBe(false);
    expect(
      screen.getByRole("button", { name: "Pick" }).getAttribute("data-slot"),
    ).toBe("date-input-trigger");
  });
});

describe("Button slots", () => {
  it("names the button and its label", () => {
    renderButton({ children: "Save" });
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.getAttribute("data-slot")).toBe("button");
    expect(el.querySelector("[data-slot=button-label]")?.textContent).toBe(
      "Save",
    );
  });

  it("keeps a slot the caller passes", () => {
    render(
      <TooltipProvider>
        <Button data-slot="date-input-trigger">Pick</Button>
      </TooltipProvider>,
    );
    expect(
      screen.getByRole("button", { name: "Pick" }).getAttribute("data-slot"),
    ).toBe("date-input-trigger");
  });

  it("keeps its own slot inside the tooltip trigger of an icon button", () => {
    renderButton({
      variant: "icon",
      "aria-label": "Edit",
      leadingIcon: PencilEdit02Icon,
      children: null,
    });
    const el = screen.getByRole("button", { name: "Edit" });
    expect(el.getAttribute("data-slot")).toBe("button");
    expect(el.querySelector("[data-slot=button-icon]")).not.toBeNull();
  });
});

describe("Button textSize", () => {
  it("keeps the size's text-body by default", () => {
    renderButton({ size: "small", children: "Save" });
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("text-body");
    expect(el.className).not.toContain("text-body-md");
  });

  it.each([
    ["small", "button"],
    ["large", "button"],
    ["small", "loading"],
    ["large", "loading"],
    ["small", "NextLink"],
    ["large", "NextLink"],
  ] as const)(
    "replaces the %s size's text-body and leading with text-body-md (%s)",
    (size, kind) => {
      renderButton({
        size,
        textSize: "body-md",
        children: "Save",
        ...(kind === "loading" ? { loading: true } : {}),
        ...(kind === "NextLink" ? { as: "NextLink", href: "/x" } : {}),
      } as Partial<ButtonProps>);
      const el =
        kind === "NextLink"
          ? screen.getByRole("link", { name: "Save" })
          : screen.getByRole("button", { name: "Save" });
      const classes = el.className.split(" ");
      expect(classes).toContain("text-body-md");
      for (const stale of ["text-body", "leading-5", "leading-snug"])
        expect(classes).not.toContain(stale);
    },
  );

  it("replaces text-body with text-body-md on the link variant", () => {
    renderButton({ variant: "link", textSize: "body-md", children: "Save" });
    const classes = screen
      .getByRole("button", { name: "Save" })
      .className.split(" ");
    expect(classes).toContain("text-body-md");
    expect(classes).not.toContain("text-body");
  });
});
