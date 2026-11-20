import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { render, screen, cleanup } from "@testing-library/react";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { afterEach, describe, expect, it } from "vitest";
import { tones } from "@/components/styles/tones";
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
    expect(el.className).not.toContain("border-foreground-body");
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

describe("Button compact variants", () => {
  it("renders a rounded-md button with the children as they are", () => {
    render(
      <Button variant="compact-outline" size="compact" className="space-x-3">
        <span>One</span>
        <span>Two</span>
      </Button>,
    );
    const el = screen.getByRole("button", { name: "One Two" });
    expect(el.children).toHaveLength(2);
    expect(el.querySelector(".contents")).toBeNull();
    expect(el.className).toContain("rounded-md");
    expect(el.className).not.toContain("rounded-full");
    expect(el.className).toContain("border-border-base");
    expect(el.className).toContain("h-9 px-4 py-2");
    expect(el.className).toContain("space-x-3");
  });

  it("applies the size, and lets the caller's classes win", () => {
    render(
      <Button variant="compact-primary" size="compact-sm" className="h-10">
        Save
      </Button>,
    );
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("bg-action");
    expect(el.className).toContain("h-10");
    expect(el.className).not.toContain("h-8");
    expect(el.className).toContain("text-caption");
  });

  it("forwards the ref and the caller's props", () => {
    let node: HTMLButtonElement | null = null;
    render(
      <Button
        variant="compact-ghost"
        size="compact"
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
    expect(el.getAttribute("data-slot")).toBe("button");
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

describe("Button quiet variants, row and inline sizes, and shape", () => {
  it.each([
    ["ghost-muted", "text-foreground-tertiary", "hover:text-foreground-body"],
    ["ghost-subtle", "text-foreground-subtle", "hover:text-foreground-body"],
    ["ghost-danger", "text-foreground-tertiary", "hover:text-error"],
  ] as const)("renders %s with a ghost box", (variant, rest, hover) => {
    renderButton({ variant, size: "icon-xs", "aria-label": variant });
    const classes = screen
      .getByRole("button", { name: variant })
      .className.split(" ");
    expect(classes).toEqual(
      expect.arrayContaining([
        "bg-transparent",
        "border-transparent",
        rest,
        hover,
      ]),
    );
    expect(classes).not.toContain("text-foreground-base");
  });

  it("renders row as a 32px button with no minimum width", () => {
    renderButton({ variant: "secondary", size: "row", children: "Retry" });
    const classes = screen
      .getByRole("button", { name: "Retry" })
      .className.split(" ");
    expect(classes).toEqual(expect.arrayContaining(["h-8", "min-w-0", "px-3"]));
    expect(classes).not.toContain("h-9");
  });

  it("renders a link at the inline size with no box", () => {
    renderButton({ variant: "link", size: "inline-xs", children: "Details" });
    const classes = screen
      .getByRole("button", { name: "Details" })
      .className.split(" ");
    expect(classes).toEqual(
      expect.arrayContaining(["h-auto", "p-0", "text-caption", "leading-4"]),
    );
    expect(classes).not.toContain("h-[2.875rem]");
  });

  it.each([
    ["pill", "rounded-full"],
    ["rounded", "rounded-xl"],
    ["soft", "rounded-lg"],
  ] as const)("renders shape %s over the size's radius", (shape, cls) => {
    renderButton({ variant: "primary", size: "xs", shape, children: "Go" });
    const classes = screen
      .getByRole("button", { name: "Go" })
      .className.split(" ");
    expect(classes).toContain(cls);
    expect(classes).not.toContain("rounded-md");
  });

  it("keeps a ghost-* variant's look while loading", () => {
    renderButton({ variant: "ghost-muted", loading: true, children: "Saving" });
    const classes = screen
      .getByRole("button", { name: "Saving" })
      .className.split(" ");
    expect(classes).toContain("text-foreground-tertiary");
    expect(classes).not.toContain("bg-foreground-tertiary");
  });
});

describe("Button fill variants, tone and icon-2xs", () => {
  it.each([
    [
      "brand",
      ["bg-brand", "text-action-foreground", "hover:bg-accent-strong"],
      "bg-action",
    ],
    [
      "warning",
      [
        "bg-warning-foreground",
        "text-action-foreground",
        "hover:bg-warning-foreground/90",
      ],
      "bg-action",
    ],
    [
      "success",
      ["bg-success/30", "border-success", "text-foreground-base"],
      "bg-surface",
    ],
    [
      "tray",
      [
        "bg-transparent",
        "border-transparent",
        "hover:bg-surface-hover",
        "disabled:opacity-40",
      ],
      "shadow-sm",
    ],
    [
      "ghost-warning",
      ["text-warning-foreground", "hover:bg-warning-surface"],
      "text-foreground-base",
    ],
  ] as const)(
    "renders the %s variant's own classes",
    (variant, defining, conflicting) => {
      renderButton({ variant, size: "small", children: variant });
      const classes = screen
        .getByRole("button", { name: variant })
        .className.split(" ");
      expect(classes).toEqual(expect.arrayContaining([...defining]));
      expect(classes).not.toContain(conflicting);
    },
  );

  it.each(Object.entries(tones).filter(([tone]) => tone !== "base"))(
    "replaces the variant's text colour with the %s tone",
    (tone, cls) => {
      renderButton({
        variant: "ghost",
        size: "small",
        tone: tone as keyof typeof tones,
        children: "Toned",
      });
      const classes = screen
        .getByRole("button", { name: "Toned" })
        .className.split(" ");
      expect(classes).toContain(cls);
      expect(classes).not.toContain("text-foreground-base");
    },
  );

  it("renders icon-2xs as a 24px icon-only button", () => {
    renderButton({
      variant: "ghost-subtle",
      size: "icon-2xs",
      leadingIcon: PencilEdit02Icon,
      "aria-label": "Rename",
    });
    const el = screen.getByRole("button", { name: "Rename" });
    const classes = el.className.split(" ");
    expect(classes).toEqual(
      expect.arrayContaining(["size-6", "min-w-0", "p-0"]),
    );
    expect(classes).not.toContain("h-[2.875rem]");
    expect(el.querySelector("svg")?.getAttribute("width")).toBe("14");
  });
});
