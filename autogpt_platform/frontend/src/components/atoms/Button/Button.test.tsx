import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { render, screen, cleanup } from "@testing-library/react";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { createRef } from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { Button } from "./Button";
import { ButtonProps } from "./helpers";

// The shared setup mocks next/link with a component that drops refs.
vi.mock("next/link", async () => await vi.importActual("next/link"));

// Kobra's Spinner animates its arc with the Web Animations API, which
// happy-dom does not implement.
beforeAll(() => {
  if (typeof Element.prototype.animate === "function") return;
  Element.prototype.animate = vi.fn(
    () =>
      ({
        onfinish: null,
        playState: "running",
        cancel: vi.fn(),
        pause: vi.fn(),
        play: vi.fn(),
      }) as unknown as Animation,
  );
  Element.prototype.getAnimations = vi.fn(() => []);
});

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

describe("Button unmask prop", () => {
  it("applies sentry-unmask class by default", () => {
    renderButton({ children: "Save" });
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("omits sentry-unmask class when unmask is false", () => {
    renderButton({ unmask: false, children: "Dynamic label" });
    const el = screen.getByRole("button", { name: "Dynamic label" });
    expect(el.className).not.toContain("sentry-unmask");
  });

  it("applies sentry-unmask when unmask is explicitly true", () => {
    renderButton({ unmask: true, children: "Explicit" });
    const el = screen.getByRole("button", { name: "Explicit" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("applies sentry-unmask to link variant button", () => {
    renderButton({ variant: "link", children: "Link button" });
    const el = screen.getByRole("button", { name: "Link button" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("omits sentry-unmask from link variant when unmask is false", () => {
    renderButton({
      variant: "link",
      unmask: false,
      children: "Dynamic link",
    });
    const el = screen.getByRole("button", { name: "Dynamic link" });
    expect(el.className).not.toContain("sentry-unmask");
  });

  it("applies sentry-unmask to ghost variant", () => {
    renderButton({ variant: "ghost", children: "Ghost" });
    const el = screen.getByRole("button", { name: "Ghost" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("applies sentry-unmask to secondary variant", () => {
    renderButton({ variant: "secondary", children: "Secondary" });
    const el = screen.getByRole("button", { name: "Secondary" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("preserves custom className alongside sentry-unmask", () => {
    renderButton({ className: "my-class", children: "Styled" });
    const el = screen.getByRole("button", { name: "Styled" });
    expect(el.className).toContain("sentry-unmask");
    expect(el.className).toContain("my-class");
  });

  it("applies sentry-unmask in the loading state", () => {
    renderButton({ loading: true, children: "Saving" });
    const el = screen.getByRole("button");
    expect(el.className).toContain("sentry-unmask");
  });

  it("omits sentry-unmask in the loading state when unmask is false", () => {
    renderButton({ loading: true, unmask: false, children: "Saving" });
    const el = screen.getByRole("button");
    expect(el.className).not.toContain("sentry-unmask");
  });

  it("applies sentry-unmask to NextLink buttons", () => {
    renderButton({ as: "NextLink", href: "/save", children: "Go" });
    const el = screen.getByRole("link", { name: "Go" });
    expect(el.className).toContain("sentry-unmask");
  });

  it("omits sentry-unmask from NextLink buttons when unmask is false", () => {
    renderButton({
      as: "NextLink",
      href: "/save",
      unmask: false,
      children: "Dynamic NextLink",
    });
    const el = screen.getByRole("link", { name: "Dynamic NextLink" });
    expect(el.className).not.toContain("sentry-unmask");
  });
});

describe("Button sizes", () => {
  it.each([
    ["sm", "h-8"],
    ["md", "h-9"],
    ["lg", "h-10"],
  ] as const)("renders %s at %s", (size, height) => {
    renderButton({ size, children: "Save" });
    expect(screen.getByRole("button", { name: "Save" }).className).toContain(
      height,
    );
  });

  it("defaults to lg", () => {
    renderButton({ children: "Save" });
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("h-10");
    expect(el.className).toContain("min-w-30");
  });

  it("renders sm as a rounded-rectangle chip without a minimum width", () => {
    renderButton({ variant: "secondary", size: "sm", children: "Chat" });
    const el = screen.getByRole("button", { name: "Chat" });
    expect(el.className).toContain("rounded-md");
    expect(el.className).not.toContain("rounded-full");
    expect(el.className).not.toContain("min-w-");
  });

  it("renders a leadingIcon before the label", () => {
    renderButton({
      size: "sm",
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
      size: "icon-sm",
      leadingIcon: PencilEdit02Icon,
      "aria-label": "Edit workflow",
      children: undefined,
    });
    const el = screen.getByRole("button", { name: "Edit workflow" });
    expect(el.className).toContain("size-8");
    expect(el.className).toContain("bg-card/90");
  });

  it("styles toggle by aria-pressed", () => {
    renderButton({
      variant: "toggle",
      size: "sm",
      "aria-pressed": true,
      children: "Needs review",
    });
    const el = screen.getByRole("button", {
      name: "Needs review",
      pressed: true,
    });
    expect(el.className).toContain("aria-pressed:bg-muted");
  });
});

describe("Button tokens", () => {
  it("paints primary with Kobra's primary surface", () => {
    renderButton({ children: "Save" });
    const el = screen.getByRole("button", { name: "Save" });
    expect(el.className).toContain("t-surface-primary");
    expect(el.className).toContain("text-primary-foreground");
    expect(el.className).toContain("rounded-full");
    expect(el.getAttribute("data-slot")).toBe("button");
  });

  it("uses Kobra's focus ring", () => {
    renderButton({ variant: "secondary", children: "Cancel" });
    expect(screen.getByRole("button", { name: "Cancel" }).className).toContain(
      "focus-visible:ring-ring/50",
    );
  });

  it("keeps the variant while loading instead of turning grey", () => {
    renderButton({ variant: "secondary", loading: true, children: "Saving" });
    const el = screen.getByRole("button", { name: "Saving" });
    expect(el.className).toContain("t-surface-outline");
    expect(el).toHaveProperty("disabled", true);
    expect(el.getAttribute("aria-busy")).toBe("true");
    expect(el.querySelector("[data-slot=spinner]")).not.toBeNull();
  });

  it("defaults to type=button and keeps an explicit submit", () => {
    renderButton({ children: "Default" });
    expect(
      screen.getByRole("button", { name: "Default" }).getAttribute("type"),
    ).toBe("button");
    cleanup();
    renderButton({ type: "submit", children: "Submit" });
    expect(
      screen.getByRole("button", { name: "Submit" }).getAttribute("type"),
    ).toBe("submit");
  });

  it("renders the sm chip through Kobra at the house height", () => {
    renderButton({ variant: "primary", size: "sm", children: "Chat" });
    const el = screen.getByRole("button", { name: "Chat" });
    expect(el.className).toContain("h-8");
    expect(el.className).not.toContain("h-7");
  });
});

describe("Button ref forwarding", () => {
  it("forwards the ref to the button element", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Save</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    expect(ref.current).toBe(screen.getByRole("button", { name: "Save" }));
  });

  it("forwards the ref through the icon tooltip wrapper", () => {
    const ref = createRef<HTMLButtonElement>();
    render(
      <TooltipProvider>
        <Button
          ref={ref}
          variant="icon"
          aria-label="Edit"
          leadingIcon={PencilEdit02Icon}
        />
      </TooltipProvider>,
    );
    expect(ref.current).toBe(screen.getByRole("button", { name: "Edit" }));
  });

  it("forwards the ref to the anchor element for NextLink", () => {
    const ref = createRef<HTMLAnchorElement>();
    render(
      <Button ref={ref} as="NextLink" href="/library">
        Library
      </Button>,
    );
    expect(ref.current).toBeInstanceOf(HTMLAnchorElement);
    expect(ref.current).toBe(screen.getByRole("link", { name: "Library" }));
    expect(ref.current?.getAttribute("href")).toBe("/library");
    expect(ref.current?.getAttribute("role")).toBeNull();
  });

  it("marks a disabled NextLink with aria-disabled", () => {
    render(
      <Button as="NextLink" href="/library" disabled>
        Library
      </Button>,
    );
    const el = screen.getByRole("link", { name: "Library" });
    expect(el.getAttribute("aria-disabled")).toBe("true");
    expect(el.className).toContain("pointer-events-none");
  });
});
