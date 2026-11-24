import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { Button } from "@/components/atoms/Button/Button";
import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { tones } from "@/components/styles/tones";
import { cn } from "@/lib/utils";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { cleanup, render } from "@testing-library/react";
import type React from "react";
import { afterEach, describe, expect, it } from "vitest";
import {
  OldButton,
  OLD_SHAPE_NAMES,
  OLD_SIZE_NAMES,
  OLD_VARIANT_NAMES,
  oldButtonVariants,
  type OldButtonProps,
} from "./button-variant-map.fixture";
import { migrateButtonStyle, type OldButtonStyle } from "./button-variant-map";

afterEach(() => {
  cleanup();
});

// Every (variant, size, shape, tone) a call site in src/ passed at 7fda55e2bc,
// read from the JSX with the TypeScript compiler: both branches of each
// conditional, and the values the wrappers that forward a variant or size
// (ExpertWorkflowActions, PlanChoice, UnblockDialog, InputGroup, Message,
// IconButton, TopUpForm, PaymentDueAction) are given.
const USED: OldButtonStyle[] = [
  {},
  { size: "large" },
  { size: "small" },
  { size: "small", shape: "soft" },
  { variant: "brand", size: "small", shape: "soft" },
  { variant: "compact-destructive", size: "compact" },
  { variant: "compact-ghost", size: "compact" },
  { variant: "compact-ghost", size: "compact-icon" },
  { variant: "compact-ghost", size: "compact-icon-sm" },
  { variant: "compact-ghost", size: "compact-sm" },
  { variant: "compact-outline", size: "compact" },
  { variant: "compact-outline", size: "compact-icon" },
  { variant: "compact-outline", size: "compact-sm" },
  { variant: "compact-primary", size: "compact" },
  { variant: "compact-primary", size: "compact-icon" },
  { variant: "compact-primary", size: "compact-icon-sm" },
  { variant: "destructive" },
  { variant: "destructive", size: "icon" },
  { variant: "destructive", size: "icon-sm" },
  { variant: "destructive", size: "icon-sm", shape: "pill" },
  { variant: "destructive", size: "small" },
  { variant: "floating", size: "icon-xs" },
  { variant: "ghost" },
  { variant: "ghost", shape: "rounded" },
  { variant: "ghost", size: "icon" },
  { variant: "ghost", size: "icon-2xs", shape: "pill" },
  { variant: "ghost", size: "icon-sm" },
  { variant: "ghost", size: "icon-sm", tone: "muted" },
  { variant: "ghost", size: "icon-sm", shape: "pill" },
  { variant: "ghost", size: "icon-xs" },
  { variant: "ghost", size: "icon-xs", tone: "inherit" },
  { variant: "ghost", size: "inline" },
  { variant: "ghost", size: "inline", tone: "accent" },
  { variant: "ghost", size: "inline", tone: "emphasis" },
  { variant: "ghost", size: "inline", tone: "secondary" },
  { variant: "ghost", size: "large" },
  { variant: "ghost", size: "row" },
  { variant: "ghost", size: "row", tone: "secondary" },
  { variant: "ghost", size: "small" },
  { variant: "ghost", size: "small", tone: "brand" },
  { variant: "ghost", size: "small", tone: "info-strong" },
  { variant: "ghost", size: "small", tone: "secondary" },
  { variant: "ghost", size: "small", shape: "pill" },
  { variant: "ghost", size: "small", shape: "rounded" },
  { variant: "ghost", size: "small", shape: "soft" },
  { variant: "ghost", size: "xs" },
  { variant: "ghost", size: "xs", tone: "secondary" },
  { variant: "ghost", size: "xs", shape: "pill" },
  { variant: "ghost-danger", size: "icon-xs", shape: "soft" },
  { variant: "ghost-muted", size: "icon-sm" },
  { variant: "ghost-muted", size: "icon-sm", shape: "pill" },
  { variant: "ghost-muted", size: "icon-xs", shape: "soft" },
  { variant: "ghost-muted", size: "inline-xs" },
  { variant: "ghost-muted", size: "small" },
  { variant: "ghost-muted", size: "small", shape: "pill" },
  { variant: "ghost-subtle", size: "icon-2xs", shape: "soft" },
  { variant: "ghost-warning" },
  { variant: "ghost-warning", size: "icon" },
  { variant: "icon", size: "icon" },
  { variant: "icon", size: "icon", tone: "subtle" },
  { variant: "icon", size: "icon-sm" },
  { variant: "icon", size: "icon-sm", shape: "pill" },
  { variant: "icon", size: "icon-xs" },
  { variant: "icon", size: "small" },
  { variant: "link" },
  { variant: "link", tone: "emphasis" },
  { variant: "link", tone: "secondary" },
  { variant: "link", size: "inline" },
  { variant: "link", size: "inline-xs" },
  { variant: "link", size: "inline-xs", tone: "brand" },
  { variant: "link", size: "inline-xs", tone: "muted" },
  { variant: "link", size: "small" },
  { variant: "link", size: "small", tone: "muted" },
  { variant: "outline" },
  { variant: "outline", size: "icon" },
  { variant: "outline", size: "icon-xs", shape: "pill" },
  { variant: "outline", size: "large" },
  { variant: "outline", size: "small" },
  { variant: "outline", size: "small", tone: "info-strong" },
  { variant: "outline", size: "small", tone: "warning" },
  { variant: "outline", size: "small", shape: "soft" },
  { variant: "outline", size: "xs" },
  { variant: "outline", size: "xs", tone: "danger" },
  { variant: "primary" },
  { variant: "primary", size: "icon" },
  { variant: "primary", size: "icon-2xs", shape: "pill" },
  { variant: "primary", size: "icon-sm" },
  { variant: "primary", size: "icon-sm", shape: "pill" },
  { variant: "primary", size: "inline-xs" },
  { variant: "primary", size: "inline-xs", tone: "muted" },
  { variant: "primary", size: "large" },
  { variant: "primary", size: "row" },
  { variant: "primary", size: "small" },
  { variant: "primary", size: "small", tone: "muted" },
  { variant: "primary", size: "small", tone: "primary" },
  { variant: "primary", size: "small", tone: "warning" },
  { variant: "primary", size: "small", shape: "rounded" },
  { variant: "primary", size: "xs" },
  { variant: "primary", size: "xs", shape: "pill" },
  { variant: "secondary" },
  { variant: "secondary", size: "icon" },
  { variant: "secondary", size: "icon-2xs", shape: "pill" },
  { variant: "secondary", size: "icon-sm" },
  { variant: "secondary", size: "icon-sm", shape: "pill", tone: "base" },
  { variant: "secondary", size: "icon-xs", shape: "pill" },
  { variant: "secondary", size: "inline-xs" },
  { variant: "secondary", size: "inline-xs", tone: "muted" },
  { variant: "secondary", size: "large" },
  { variant: "secondary", size: "row" },
  { variant: "secondary", size: "small" },
  { variant: "secondary", size: "small", tone: "danger" },
  { variant: "secondary", size: "small", tone: "muted" },
  { variant: "secondary", size: "small", shape: "rounded" },
  { variant: "secondary", size: "xs" },
  { variant: "secondary", size: "xs", shape: "pill" },
  { variant: "success", size: "icon-sm", shape: "pill", tone: "base" },
  { variant: "toggle", size: "icon-xs" },
  { variant: "toggle", size: "inline" },
  { variant: "toggle", size: "small" },
  { variant: "toggle", size: "xs" },
  { variant: "tray", size: "icon" },
  { variant: "warning", size: "small" },
];

// What a call site passes next to the style props: layout, and the restyles
// the baseline still allows, including ones that collide with the atom's.
const CALL_SITES: Record<string, unknown>[] = [
  { className: "mt-2 w-full" },
  { className: "min-w-0 h-10 size-9 px-2" },
  { className: "rounded-l-none border-r-0" },
  { className: "!rounded-r-none" },
  { className: "text-foreground-tertiary hover:text-foreground-body" },
  { className: "hover:bg-transparent bg-surface/80" },
  { className: "disabled:opacity-60" },
];

// The props that change which branch renders or which classes it adds.
const STATES: Record<string, unknown>[] = [
  {},
  { disabled: true },
  { loading: true },
  { as: "NextLink", href: "/x" },
  { as: "NextLink", href: "/x", disabled: true },
  { as: "NextLink", href: "/x", loading: true },
  { "aria-label": "Edit", leadingIcon: PencilEdit02Icon, children: undefined },
  { "aria-label": "Edit", withTooltip: false, children: undefined },
  { textSize: "body-md" },
  { asChild: true },
];

const COMPACT_STATES: Record<string, unknown>[] = [
  {},
  { disabled: true },
  { "aria-label": "Edit", children: undefined },
];

function html(element: React.ReactElement) {
  const { container } = render(<TooltipProvider>{element}</TooltipProvider>);
  const result = container.innerHTML.replace(
    /class="([^"]*)"/g,
    (_, classes: string) => `class="${classSet(classes).join(" ")}"`,
  );
  cleanup();
  return result;
}

// disabled:opacity-1 is not a Tailwind class and generated nothing; the
// new atom no longer writes it.
function classSet(className: string) {
  return className
    .split(/\s+/)
    .filter((name) => name && name !== "disabled:opacity-1")
    .sort();
}

function compare(old: OldButtonStyle, extra: Record<string, unknown>) {
  const migration = migrateButtonStyle(old);
  if (!("props" in migration)) throw new Error(migration.conflict);
  const oldProps = { children: "Label", ...old, ...extra };
  const newProps = { ...oldProps, ...migration.props };
  const label = `${JSON.stringify(old)} + ${JSON.stringify(extra)}`;
  expect(
    html(<Button {...(newProps as React.ComponentProps<typeof Button>)} />),
    label,
  ).toBe(html(<OldButton {...(oldProps as OldButtonProps)} />));
}

function isCompact(style: OldButtonStyle) {
  return style.variant?.startsWith("compact-") ?? false;
}

describe("migrateButtonStyle", () => {
  it.each(USED.map((style) => [JSON.stringify(style), style] as const))(
    "renders %s as before in every state and with every call site",
    (_, style) => {
      for (const extra of isCompact(style)
        ? COMPACT_STATES
        : [...STATES, ...CALL_SITES])
        compare(style, extra);
    },
  );

  it("renders every variant, size and shape as before", () => {
    const sizes = [undefined, ...OLD_SIZE_NAMES];
    const shapes = [undefined, ...OLD_SHAPE_NAMES];
    expect(OLD_VARIANT_NAMES).toHaveLength(17);
    expect(sizes).toHaveLength(11);
    for (const variant of OLD_VARIANT_NAMES.filter((name) => name !== "link"))
      for (const size of sizes)
        for (const shape of shapes)
          for (const extra of [{}, { loading: true }, { className: "w-full" }])
            compare({ variant, size, shape }, extra);
    for (const size of sizes)
      for (const shape of shapes) compare({ variant: "link", size, shape }, {});
  });

  it("renders every tone on every variant as before", () => {
    for (const variant of OLD_VARIANT_NAMES.filter(
      (name) => !name.startsWith("ghost-"),
    ))
      for (const tone of Object.keys(tones))
        compare({ variant, size: "small", tone }, {});
  });

  it("refuses a quiet ghost variant that also sets a tone", () => {
    expect(
      migrateButtonStyle({ variant: "ghost-muted", tone: "base" }),
    ).toEqual({ conflict: expect.stringContaining("ghost-muted") });
  });

  // NewMenu styles a menu trigger with the helper itself.
  it("keeps the classes NewMenu builds from the helper", () => {
    expect(
      classSet(
        cn(extendedButtonVariants({ variant: "primary", size: "small" })),
      ),
    ).toEqual(
      classSet(cn(oldButtonVariants({ variant: "primary", size: "small" }))),
    );
  });
});
