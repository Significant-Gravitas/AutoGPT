import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { render, screen, cleanup } from "@testing-library/react";
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
