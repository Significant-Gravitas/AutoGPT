import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Text, type TextProps } from "@/components/atoms/Text/Text";
import { textModifiers } from "@/components/atoms/Text/helpers";
import { cn } from "@/lib/utils";
import { OLD_TEXT_VARIANTS } from "./text-variant-map.fixture";
import {
  TEXT_VARIANT_MAP,
  type OldTextVariant,
  type TextVariantMapping,
} from "./text-variant-map";

afterEach(() => {
  cleanup();
});

type CallSite = Omit<Partial<TextProps>, "variant" | "size" | "children">;

// What call sites pass next to a variant: each prop the old atom applied
// after the variant's classes, and className overrides that collide with them.
const CALL_SITES: CallSite[] = [
  {},
  { weight: "normal" },
  { weight: "medium" },
  { weight: "semibold" },
  { weight: "bold" },
  { tone: "muted" },
  { tone: "inherit" },
  { mono: true },
  { italic: true },
  { tabular: true },
  { decoration: "underline" },
  { decoration: "line-through" },
  { decoration: "none" },
  { transform: "uppercase" },
  { transform: "none" },
  { tracking: "tight" },
  { tracking: "caps" },
  { align: "center" },
  { motion: "colors" },
  { as: "span" },
  { as: "label" },
  { className: "mt-2 truncate" },
  { className: "leading-6" },
  { className: "text-body" },
  { className: "font-normal" },
  { className: "tracking-tight" },
  { weight: "semibold", tone: "muted", className: "truncate" },
];

const OLD_NAMES = Object.keys(TEXT_VARIANT_MAP) as OldTextVariant[];

function oldOutput(variant: string, callSite: CallSite, size?: string) {
  const { as, className, ...modifiers } = callSite;
  return {
    classes: classSet(
      cn(
        OLD_TEXT_VARIANTS[size ?? variant].classes,
        textModifiers(modifiers),
        className,
      ),
    ),
    tag: (as ?? OLD_TEXT_VARIANTS[variant].element).toUpperCase(),
  };
}

function newOutput(
  variant: OldTextVariant,
  callSite: CallSite,
  size?: OldTextVariant,
) {
  const element: TextVariantMapping = TEXT_VARIANT_MAP[variant];
  const classSource: TextVariantMapping = TEXT_VARIANT_MAP[size ?? variant];
  const { container } = render(
    <Text
      variant={element.variant}
      size={size ? classSource.variant : undefined}
      as={element.as}
      {...classSource.props}
      {...callSite}
    >
      x
    </Text>,
  );
  const node = container.firstElementChild as HTMLElement;
  const result = { classes: classSet(node.className), tag: node.tagName };
  cleanup();
  return result;
}

function classSet(className: string) {
  return className.split(/\s+/).filter(Boolean).sort();
}

describe("TEXT_VARIANT_MAP", () => {
  it("covers every old variant", () => {
    expect(OLD_NAMES.sort()).toEqual(Object.keys(OLD_TEXT_VARIANTS).sort());
    expect(OLD_NAMES).toHaveLength(67);
  });

  it.each(OLD_NAMES)(
    "renders %s with the old classes and element under every call site",
    (variant) => {
      for (const callSite of CALL_SITES) {
        expect(
          newOutput(variant, callSite),
          `${variant} + ${JSON.stringify(callSite)}`,
        ).toEqual(oldOutput(variant, callSite));
      }
    },
  );

  // `size` took a variant's classes while `variant` kept picking the element.
  it.each(OLD_NAMES)("renders %s as size under h3 and body", (size) => {
    for (const variant of ["h3", "body", "small-mono-tight"] as const) {
      for (const callSite of [
        {},
        { weight: "bold" as const },
        { as: "span" as const },
      ]) {
        expect(
          newOutput(variant, callSite, size),
          `${variant} size=${size} + ${JSON.stringify(callSite)}`,
        ).toEqual(oldOutput(variant, callSite, size));
      }
    }
  });
});
