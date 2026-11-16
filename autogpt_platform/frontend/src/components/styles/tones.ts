/**
 * Semantic colours shared by the `Text` and `Icon` atoms (`tone` prop). Each
 * is a colour role from src/app/tokens/theme.css, noted at the hex it renders in
 * the light theme. Text without a tone keeps its variant's colour
 * (foreground-base #141414); an Icon without one takes the colour around it.
 */
export const tones = {
  base: "text-foreground-base", // #141414
  primary: "text-foreground-strong", // #2C2C30
  emphasis: "text-foreground-emphasis", // #3E3E43
  // emphasis at 90%: the sidebar's navigation icons.
  "emphasis-soft": "text-foreground-emphasis/90",
  body: "text-foreground-body", // #505057
  secondary: "text-foreground-secondary", // #68686F
  muted: "text-foreground-tertiary", // #83838C
  subtle: "text-foreground-subtle", // #ADADB3
  disabled: "text-foreground-disabled", // #C5C5C9
  // Text and icons on a dark fill: the primary Button, a brand chip.
  inverse: "text-action-foreground", // #FEFEFE
  brand: "text-brand", // #7733F5
  "brand-strong": "text-brand-hover", // #6C2EDF
  accent: "text-accent-strong", // #6D28D9
  danger: "text-error-hover", // #D93636
  error: "text-error", // #EF4444
  "error-strong": "text-error-foreground", // #AA3030
  warning: "text-warning-foreground", // #B45309
  "warning-strong": "text-warning-foreground", // #B45309
  // The amber of a warning icon, brighter than warning text.
  "warning-bright": "text-warning", // #F59E0B
  success: "text-success", // #149443
  "success-strong": "text-success-foreground", // #107435
  info: "text-info", // #3B82F6
  "info-strong": "text-info-foreground", // #2563EB
  // The parent's colour, for copy inside a control that changes it on hover.
  inherit: "text-inherit",
  // Green on a dark fill (toast, terminal), where `success` would lose its
  // contrast.
  "success-light": "text-success-inverse",
} as const;

export type Tone = keyof typeof tones;
