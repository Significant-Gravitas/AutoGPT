import type { FC, SVGProps } from "react";

// Props exposed by every `@autogpt/icons` component.
export interface AutoGPTIconProps extends SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  ariaLabel?: string;
}

export type AutoGPTIconComponent = FC<AutoGPTIconProps>;

export function getAutoGPTIcon(
  exportName: string,
): AutoGPTIconComponent | undefined;
