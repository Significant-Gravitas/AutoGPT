"use client";

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import type { ComponentPropsWithoutRef } from "react";

export const ICON_STROKE_WIDTH = 2;

type IconProps = Omit<ComponentPropsWithoutRef<typeof HugeiconsIcon>, "icon">;

interface Props extends IconProps {
  icon: IconSvgElement;
}

export function Icon({
  size = "1em",
  strokeWidth = ICON_STROKE_WIDTH,
  ...props
}: Props) {
  return <HugeiconsIcon size={size} strokeWidth={strokeWidth} {...props} />;
}

/**
 * Wraps icon data in a component, for maps that also hold icons from other
 * libraries (react-icons, radix) and therefore need a uniform component type.
 */
export function createIconComponent(icon: IconSvgElement) {
  return function IconComponent(props: IconProps) {
    return <Icon icon={icon} {...props} />;
  };
}
