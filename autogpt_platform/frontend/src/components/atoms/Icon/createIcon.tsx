import { forwardRef } from "react";
import type { Icon as PhosphorIcon, IconProps } from "@phosphor-icons/react";
import { getAutoGPTIcon, type AutoGPTIconProps } from "./agptIcons";
import { areAutoGPTIconsAvailable } from "./helpers";
import { iconFillMap } from "./iconFillMap";

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
      // `weight`/`mirrored` are Phosphor-only, so drop them before spreading
      // onto the AutoGPT <svg>. weight="fill" swaps to the solid variant when
      // the set has one (see iconFillMap) so filled-state toggles stay
      // visible; every other weight renders the fixed stroke style. Pull out
      // `aria-label` too — it's re-applied below via the `ariaLabel` prop, and
      // spreading it as well would set the attribute twice.
      const {
        weight,
        mirrored: _mirrored,
        "aria-label": label,
        ...rest
      } = props;
      const fillExport =
        weight === "fill" ? iconFillMap[autogptExport] : undefined;
      const AutoGPTIcon = getAutoGPTIcon(fillExport ?? autogptExport);
      if (AutoGPTIcon) {
        // Phosphor and AutoGPT icons take the same runtime SVG props; the cast
        // reconciles Phosphor's `size: string | number` with AutoGPT's numeric.
        const autogptProps = rest as unknown as AutoGPTIconProps;
        if (typeof label === "string") {
          return <AutoGPTIcon {...autogptProps} ariaLabel={label} />;
        }
        // No explicit label: render decorative to match Phosphor's default, so
        // the icon never pollutes the accessible name of its parent. AutoGPT
        // icons are decorative-by-default since 0.3.0; force the contract
        // explicitly anyway so it holds regardless of the installed version
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
