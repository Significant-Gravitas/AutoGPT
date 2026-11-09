import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { translateText } from "@/tests/integrations/translation";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { Button } from "./Button";
import type { ButtonProps } from "./helpers";

afterEach(cleanup);

const cases: Partial<ButtonProps>[] = [
  { variant: "primary" },
  { variant: "secondary" },
  { variant: "ghost" },
  { variant: "link" },
  { as: "NextLink", href: "/onboarding" },
];

it.each(cases)("preserves translated labels through loading: %j", (props) => {
  function TestButton({ loading }: { loading: boolean }) {
    return (
      <TooltipProvider>
        <Button
          {...(props as ButtonProps)}
          loading={loading}
          data-goal="checkout"
        >
          Get Pro
        </Button>
      </TooltipProvider>
    );
  }

  const { rerender } = render(<TestButton loading={false} />);
  const role = props.as === "NextLink" ? "link" : "button";
  const button = screen.getByRole(role, { name: "Get Pro" });
  const translation = translateText(button, "Get Pro", "احصل على برو");

  rerender(<TestButton loading />);

  expect(screen.getByRole(role, { name: "احصل على برو" })).toBe(button);
  expect(button.contains(translation)).toBe(true);
  expect(button.getAttribute("data-goal")).toBe("checkout");
  expect(
    role === "link"
      ? button.getAttribute("aria-disabled") === "true"
      : button.hasAttribute("disabled"),
  ).toBe(true);

  rerender(<TestButton loading={false} />);

  expect(screen.getByRole(role, { name: "احصل على برو" })).toBe(button);
  expect(button.contains(translation)).toBe(true);
  expect(button.hasAttribute("disabled")).toBe(false);
  expect(button.getAttribute("aria-disabled")).not.toBe("true");
});

it("can finish loading after the page is translated during the request", () => {
  const { rerender } = render(<Button loading>Get Pro</Button>);
  const button = screen.getByRole("button", { name: "Get Pro" });
  translateText(button, "Get Pro", "احصل على برو");

  rerender(<Button>Get Pro</Button>);

  expect(screen.getByRole("button", { name: "احصل على برو" })).toBe(button);
  expect(button.hasAttribute("disabled")).toBe(false);
});
