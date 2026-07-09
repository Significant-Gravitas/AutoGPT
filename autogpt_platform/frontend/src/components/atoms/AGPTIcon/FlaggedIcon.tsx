"use client";

import type { ComponentType, SVGProps } from "react";

import type { Icon as PhosphorIcon, IconProps } from "@phosphor-icons/react";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useIconSetStore } from "@/services/icon-set/useIconSet";

type PikaIconProps = {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
} & Omit<SVGProps<SVGSVGElement>, "size">;

export type PikaIcon = ComponentType<PikaIconProps>;

interface Props extends IconProps {
  phosphor: PhosphorIcon;
  pika: PikaIcon;
}

/**
 * Renders a Pikaicons component when the ``PIKA_ICONS`` flag is on and the
 * original Phosphor icon otherwise, so the whole app's icon set can be
 * swapped from a single flag. Phosphor-only props (``weight``, ``mirrored``,
 * ``alt``) are dropped for Pika, which has no equivalent — ``alt`` maps onto
 * the accessible label.
 */
export function FlaggedIcon({
  phosphor: Phosphor,
  pika: Pika,
  weight,
  mirrored,
  alt,
  size,
  ...rest
}: Props) {
  // The Appearance setting (localStorage) overrides the flag when the user has
  // made an explicit choice; otherwise fall back to the `PIKA_ICONS` flag.
  const flagPika = useGetFlag(Flag.PIKA_ICONS);
  const iconSet = useIconSetStore((state) => state.iconSet);
  const usePika = iconSet === null ? flagPika : iconSet === "pika";

  if (usePika) {
    // Pikaicons only accept a numeric size; drop Phosphor's ``1em``-style
    // string sizes and let the icon fall back to its own default.
    const pikaSize = typeof size === "number" ? size : undefined;
    return <Pika aria-label={alt} size={pikaSize} {...rest} />;
  }

  return (
    <Phosphor
      weight={weight}
      mirrored={mirrored}
      alt={alt}
      size={size}
      {...rest}
    />
  );
}
