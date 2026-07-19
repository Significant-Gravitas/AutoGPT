import type { IconWeight } from "@phosphor-icons/react";
import { getAutoGPTIcon } from "./agptIcons";
import { areAutoGPTIconsAvailable } from "./helpers";
import { iconRegistry, type IconName } from "./registry";

interface Props {
  name: IconName;
  size?: number;
  color?: string;
  className?: string;
  // Phosphor-only stroke weight; ignored when the AutoGPT icon is used.
  weight?: IconWeight;
  "aria-label"?: string;
}

// Renders a semantic icon. When the optional `@autogpt/icons` package is
// installed, the entire app renders AutoGPT icons; otherwise every icon falls
// back to Phosphor. The two sets are never mixed (see areAutoGPTIconsAvailable).
// Feature code should use this instead of importing icon libraries directly.
export function Icon({
  name,
  size = 24,
  color,
  className,
  weight = "regular",
  "aria-label": ariaLabel,
}: Props) {
  const entry = iconRegistry[name];
  const label = ariaLabel ?? name;

  if (areAutoGPTIconsAvailable()) {
    // Guaranteed defined once the availability check passes; the guard only
    // narrows the type for TypeScript.
    const AutoGPTIcon = getAutoGPTIcon(entry.autogpt);
    if (AutoGPTIcon) {
      return (
        <AutoGPTIcon
          size={size}
          color={color}
          className={className}
          ariaLabel={label}
        />
      );
    }
  }

  const PhosphorIcon = entry.phosphor;
  return (
    <PhosphorIcon
      size={size}
      color={color}
      className={className}
      weight={weight}
      aria-label={label}
    />
  );
}
