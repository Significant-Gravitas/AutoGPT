import React from "react";

/**
 * PiGlobeStroke icon from the stroke style in apps-&-social category.
 */
interface PiGlobeStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiGlobeStroke({
  size = 24,
  color,
  className,
  ariaLabel = "globe icon",
  ...props
}: PiGlobeStrokeProps): JSX.Element {
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
        d="M2.85 12h18.3m-18.3 0A9.15 9.15 0 0 0 12 21.15M2.85 12A9.15 9.15 0 0 1 12 2.85M21.15 12A9.15 9.15 0 0 1 12 21.15M21.15 12A9.15 9.15 0 0 0 12 2.85m0 0A14 14 0 0 1 15.66 12 14 14 0 0 1 12 21.15m0-18.3A14 14 0 0 0 8.34 12 14 14 0 0 0 12 21.15"
        fill="none"
      />
    </svg>
  );
}
