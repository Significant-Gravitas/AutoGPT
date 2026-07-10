import React from "react";

/**
 * PiPowerDefaultStroke icon from the stroke style in general category.
 */
interface PiPowerDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiPowerDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "power-default icon",
  ...props
}: PiPowerDefaultStrokeProps): JSX.Element {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: color || "currentColor" }}
      role="img"
      aria-label={ariaLabel}
      {...props}
    >
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 8V2M7.556 4a9.15 9.15 0 1 0 8.889 0"
        fill="none"
      />
    </svg>
  );
}
