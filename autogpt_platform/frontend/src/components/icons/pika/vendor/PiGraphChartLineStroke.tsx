import React from "react";

/**
 * PiGraphChartLineStroke icon from the stroke style in chart-&-graph category.
 */
interface PiGraphChartLineStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiGraphChartLineStroke({
  size = 24,
  color,
  className,
  ariaLabel = "graph-chart-line icon",
  ...props
}: PiGraphChartLineStrokeProps): JSX.Element {
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
        d="M21 21H7a4 4 0 0 1-4-4V3"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="m7 17 1.992-4.97m10.433-4.114A2 2 0 0 0 22 6a2 2 0 1 0-2.575 1.916Zm0 0-1.85 5.207m0 0a2 2 0 0 0-2.093.615m2.093-.615a2.001 2.001 0 1 1-2.093.615m0 0-3.963-2.135m0 0a2 2 0 1 0-2.526.426m2.525-.426a1.995 1.995 0 0 1-2.526.426"
        fill="none"
      />
    </svg>
  );
}
