import React from "react";

/**
 * PiFilterFunnelStroke icon from the stroke style in general category.
 */
interface PiFilterFunnelStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFilterFunnelStroke({
  size = 24,
  color,
  className,
  ariaLabel = "filter-funnel icon",
  ...props
}: PiFilterFunnelStrokeProps): JSX.Element {
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
        d="M20 4H4v2.586a1 1 0 0 0 .293.707l5.414 5.414a1 1 0 0 1 .293.707V18l4 3v-7.586a1 1 0 0 1 .293-.707l5.414-5.414A1 1 0 0 0 20 6.586z"
        fill="none"
      />
    </svg>
  );
}
