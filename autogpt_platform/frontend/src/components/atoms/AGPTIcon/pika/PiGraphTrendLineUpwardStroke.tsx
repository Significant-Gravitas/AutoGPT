import React from "react";

/**
 * PiGraphTrendLineUpwardStroke icon from the stroke style in chart-&-graph category.
 */
interface PiGraphTrendLineUpwardStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiGraphTrendLineUpwardStroke({
  size = 24,
  color,
  className,
  ariaLabel = "graph-trend-line-upward icon",
  ...props
}: PiGraphTrendLineUpwardStrokeProps): JSX.Element {
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
        d="M21 21H7a4 4 0 0 1-4-4V3m4 11 .61-1.83c.745-2.236 3.92-2.199 4.612.053.643 2.088 3.504 2.325 4.481.37l2.463-4.925m0 0c.12.06.22.16.28.291.435.962.75 1.975.938 3.016m-1.218-3.307a.6.6 0 0 0-.4-.05 13 13 0 0 0-2.979 1.05"
        fill="none"
      />
    </svg>
  );
}
