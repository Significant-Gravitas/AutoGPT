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
  return (
    <HugeiconsIcon
      data-slot="icon"
      size={size}
      strokeWidth={strokeWidth}
      {...props}
    />
  );
}

/**
 * Wraps icon data in a component, for props and maps that take a component
 * type rather than icon data (Alert's `icon`, the credential provider icons).
 */
export function createIconComponent(icon: IconSvgElement) {
  return function IconComponent(props: IconProps) {
    return <Icon icon={icon} {...props} />;
  };
}
