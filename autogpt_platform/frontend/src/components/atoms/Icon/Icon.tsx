"use client";

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { cva } from "class-variance-authority";
import type { ComponentPropsWithoutRef } from "react";
import { type Tone, tones } from "@/components/styles/tones";
import { cn } from "@/lib/utils";

export const ICON_STROKE_WIDTH = 2;

export const iconVariants = cva("", {
  variants: {
    tone: tones,
    spin: { true: "animate-spin motion-reduce:animate-none" },
    // What the icon animates when a class on it (or its group) changes: a
    // chevron's rotation, a hover colour, a reveal.
    motion: {
      transform:
        "transition-transform duration-200 motion-reduce:transition-none",
      colors: "transition-colors",
      opacity: "transition-opacity duration-200 motion-reduce:transition-none",
    },
  },
});

type IconProps = Omit<ComponentPropsWithoutRef<typeof HugeiconsIcon>, "icon">;

interface Props extends IconProps {
  icon: IconSvgElement;
  /** Colour role; without it the icon takes the colour around it. */
  tone?: Tone;
  /** Spins, e.g. a loading indicator. Stops under reduced motion. */
  spin?: boolean;
  motion?: "transform" | "colors" | "opacity";
}

export function Icon({
  size = "1em",
  strokeWidth = ICON_STROKE_WIDTH,
  tone,
  spin,
  motion,
  className,
  ...props
}: Props) {
  const variantClasses = iconVariants({ tone, spin, motion });
  return (
    <HugeiconsIcon
      data-slot="icon"
      size={size}
      strokeWidth={strokeWidth}
      className={variantClasses ? cn(variantClasses, className) : className}
      {...props}
    />
  );
}

/**
 * Wraps icon data in a component, for props and maps that take a component
 * type rather than icon data (Alert's `icon`, the credential provider icons).
 */
export function createIconComponent(icon: IconSvgElement) {
  return function IconComponent(props: Omit<Props, "icon">) {
    return <Icon icon={icon} {...props} />;
  };
}
