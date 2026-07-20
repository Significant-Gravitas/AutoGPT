import { forwardRef } from "react";
import type { Icon as PhosphorIcon, IconProps } from "@phosphor-icons/react";
import { getAutoGPTIcon, type AutoGPTIconProps } from "./agptIcons";
import { areAutoGPTIconsAvailable } from "./helpers";

// Builds a drop-in replacement for a Phosphor icon. It renders the AutoGPT icon
// when the optional @autogpt/icons package is installed (and the whole set
// resolves — see areAutoGPTIconsAvailable), otherwise the original Phosphor
// icon. The returned component accepts the same props as a Phosphor icon, so it
// can be swapped in by only changing the import source.
export function createIcon(
  autogptExport: string,
  PhosphorComponent: PhosphorIcon,
) {
  return forwardRef<SVGSVGElement, IconProps>(function Icon(props, ref) {
    if (areAutoGPTIconsAvailable()) {
      const AutoGPTIcon = getAutoGPTIcon(autogptExport);
      if (AutoGPTIcon) {
        // `weight`/`mirrored` are Phosphor-only; the AutoGPT set has a fixed
        // stroke style, so drop them before spreading onto its <svg>. Pull out
        // `aria-label` too — it's re-applied below via the `ariaLabel` prop, and
        // spreading it as well would set the attribute twice.
        const {
          weight: _weight,
          mirrored: _mirrored,
          "aria-label": label,
          ...rest
        } = props;
        // Phosphor and AutoGPT icons take the same runtime SVG props; the cast
        // reconciles Phosphor's `size: string | number` with AutoGPT's numeric.
        const autogptProps = rest as unknown as AutoGPTIconProps;
        if (typeof label === "string") {
          return <AutoGPTIcon {...autogptProps} ariaLabel={label} />;
        }
        // No explicit label: render decorative to match Phosphor's default, so
        // the icon never pollutes the accessible name of its parent. AutoGPT
        // icons hardcode `role="img"` + a default `aria-label`, so override both
        // (spread props win over the component's own attributes).
        return (
          <AutoGPTIcon
            {...autogptProps}
            role={undefined}
            aria-label={undefined}
            aria-hidden
          />
        );
      }
    }
    return <PhosphorComponent ref={ref} {...props} />;
  });
}
