/**
 * What the Button atom's old props are now. The four quiet ghost variants are
 * `ghost` with a ghost tone, and the sizes no longer carry a corner radius, so
 * a call site that relied on one (xs, icon-xs, icon-2xs and icon-sm) names
 * the matching `shape`. scripts/ds/button-variants.ts rewrites call sites
 * from these tables, and scripts/ds/button-variant-map.test.tsx renders
 * every combination in use through both atoms and proves each renders the
 * same element and class set.
 */
import type { ButtonProps } from "../../src/components/atoms/Button/helpers";

type NewVariant = NonNullable<ButtonProps["variant"]>;
type NewTone = NonNullable<ButtonProps["tone"]>;
type NewShape = NonNullable<ButtonProps["shape"]>;

export const BUTTON_VARIANT_MAP = {
  "ghost-muted": { variant: "ghost", tone: "quiet" },
  "ghost-subtle": { variant: "ghost", tone: "quiet-subtle" },
  "ghost-danger": { variant: "ghost", tone: "quiet-danger" },
  "ghost-warning": { variant: "ghost", tone: "quiet-warning" },
} as const satisfies Record<string, { variant: NewVariant; tone: NewTone }>;

/** The radius each size used to set, as the shape that sets it now. */
export const BUTTON_SIZE_SHAPE = {
  xs: "sharp",
  "icon-xs": "sharp",
  "icon-2xs": "sharp",
  "icon-sm": "soft",
} as const satisfies Record<string, NewShape>;

export interface OldButtonStyle {
  variant?: string;
  size?: string;
  shape?: string;
  tone?: string;
}

export type ButtonStyleMigration =
  | { props: OldButtonStyle }
  | { conflict: string };

function hasOwn<T extends object>(
  table: T,
  key: string,
): key is keyof T & string {
  return Object.hasOwn(table, key);
}

/**
 * The new props for an old variant, size, shape and tone. Only the props that
 * change are returned; the rest keep their value.
 */
export function migrateButtonStyle(old: OldButtonStyle): ButtonStyleMigration {
  const props: OldButtonStyle = {};
  if (old.variant !== undefined && hasOwn(BUTTON_VARIANT_MAP, old.variant)) {
    const mapped = BUTTON_VARIANT_MAP[old.variant];
    if (old.tone !== undefined)
      return {
        conflict: `${old.variant} with tone="${old.tone}": the tone would replace the quiet colour but keep its hover`,
      };
    props.variant = mapped.variant;
    props.tone = mapped.tone;
  }
  if (
    old.shape === undefined &&
    old.size !== undefined &&
    hasOwn(BUTTON_SIZE_SHAPE, old.size)
  )
    props.shape = BUTTON_SIZE_SHAPE[old.size];
  return { props };
}
